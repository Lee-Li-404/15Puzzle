from google import genai
import json, os, re, importlib.util, random, time
from typing import List, Dict, Optional, Tuple
import sys
import datetime
from google.genai import errors as genai_errors
import asyncio, itertools
from dataclasses import dataclass, field
import multiprocessing as mp
import textwrap
import ast

CTX = mp.get_context("spawn")

# ====== State ======
@dataclass
class IslandState:
    island_id: int
    results: List[Tuple[int, float, str, float, float, bool]] = field(default_factory=list)
    cnt: int = 0

# Fixed split created by sampler.py (seeded randomness)
TRAIN_FULL_FILE = "train15.txt"   # used for search/selection
TEST_FULL_FILE  = "test85.txt"    # held-out evaluation

CHECKPOINT_CSV = "best_progress.csv"
CHECKPOINT_PNG = "best_progress.png"

@dataclass
class BestEvalRecord:
    round_idx: int
    wall_time_iso: str
    island_id: int
    version: int
    score: float
    gen_train: float
    cost_train: float
    gen_test: float
    cost_test: float

BEST_HISTORY: List[BestEvalRecord] = []


# ====== Logger ======
class Logger(object):
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def safe_read_file(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] failed to read {path}: {e}")
        return None

def try_extract_heuristic(code_text: Optional[str]) -> Optional[str]:
    if not code_text:
        return None
    try:
        tree = ast.parse(code_text)
    except SyntaxError as e:
        print(f"[WARN] unparsable code skipped: {e}")
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "heuristic":
            lines = code_text.splitlines()
            start = node.lineno - 1
            end = getattr(node, "end_lineno", len(lines))
            return textwrap.dedent("\n".join(lines[start:end]))
    print("[WARN] no def heuristic found")
    return None

def rename_func(def_text: str, new_name: str) -> str:
    return re.sub(r"def\s+heuristic\s*\(", f"def {new_name}(", def_text, count=1)

def validate_api_json(text: str) -> Optional[str]:
    try:
        if not text:
            return None

        # 1) Try strict JSON first (non-greedy to avoid over-capturing)
        m = re.search(r"\{.*?\}", text, flags=re.S)
        if m:
            candidate = m.group(0)
            try:
                data = json.loads(candidate)
                if isinstance(data, dict):
                    code = data.get("code")
                    if isinstance(code, str) and code and len(code) <= MAX_CODE_CHARS:
                        if "def heuristic" in code:
                            return code
                        else:
                            print("[WARN] missing def heuristic in payload")
            except Exception as e:
                print(f"[WARN] strict JSON parse failed: {e}")

        # 2) Fallback A: markdown code fences ```python ... ```
        fence = re.search(r"```(?:python)?\s*(.+?)```", text, flags=re.S|re.I)
        if fence:
            code = fence.group(1).strip()
            if "def heuristic" in code:
                return code

        # 3) Fallback B: pull the heuristic directly from raw text
        code_from_text = try_extract_heuristic(text)
        if code_from_text:
            # ensure the required import line exists
            header = "from fifteen_state_class import State\n\n"
            full = header + code_from_text if "from fifteen_state_class import" not in text else text
            # If full is huge because it's the entire prompt, just synthesize:
            if not full.strip().startswith("from fifteen_state_class import"):
                full = header + code_from_text
            return full

        print(f"[WARN] no usable code found in model text head: {text[:200]}...")
        return None

    except Exception as e:
        print(f"[WARN] bad JSON after regex extract: {e}")
        print(f"[DEBUG] raw text head: {text[:300]}...")
        return None

# ====== Prompt builder ======
def build_best_shot_prompt(
    low_code: str, low_score: float, low_meta: Dict[str, float],
    high_code: str, high_score: float, high_meta: Dict[str, float],
    rand_list: Optional[List[Tuple[str, float]]] = None,
    prev_list: Optional[List[Tuple[str, float]]] = None
) -> str:
    """
    Build model prompt for evolving heuristics (15 Puzzle version)
    Includes expanded_ratio, cost_ratio, and global COST_BOUND.
    """
    global COST_BOUND

    low_def  = try_extract_heuristic(low_code)
    high_def = try_extract_heuristic(high_code)

    # fallback handling
    if high_def is None and low_def is not None:
        high_def, high_score, low_def, low_score = low_def, low_score, None, float("inf")
        high_meta = low_meta
    if high_def is None:
        high_def = (
            "def heuristic_v1(s: State):\n"
            "    # fallback placeholder\n"
            "    return 0\n"
        )
        high_score = float("inf")
        high_meta = {"expanded_ratio": 1.0, "cost_ratio": 1.0}

    header = (
        "You are an expert in combinatorial optimization and heuristic search algorithms.\n"
        "You are evolving a heuristic function for solving the 15 Puzzle (4×4 sliding tile puzzle) using A* search.\n"
        "\n"
        "Goal configuration:\n"
        " 0  1  2  3\n"
        " 4  5  6  7\n"
        " 8  9 10 11\n"
        "12 13 14 15\n"
        "Tile 0 is the blank tile located at the **top-left corner**.\n"
        "\n"
        "MUST start your code with:\n"
        "    from fifteen_state_class import State\n"
        "\n"
        "Reference:\n"
        "- State.tiles: tuple of 16 ints (0..15), where 0 is the blank.\n"
        "- State.neighbors() returns list of successor States.\n"
        "- State.is_goal() checks whether the puzzle is solved.\n"
        "\n"
        "Signature:\n"
        "def heuristic(s: State) -> int\n"
        "- Return a non-negative int estimate of remaining cost to goal.\n"
        "\n"
        "Rules:\n"
        "- The function does NOT need to be admissible.\n"
        "- Focus on reducing the total number of **unique nodes generated** during A* search.\n"
        f"- Keep cost_ratio ≤ {COST_BOUND:.2f}, where cost_ratio = solution_length / optimal_solution_length.\n"
        f"  This means the total solution cost (i.e., number of moves) must stay within COST_BOUND × optimal length.\n"
        "- The `cost_ratio` is the **WORST-CASE (MAXIMUM)** instance in the test set.\n"
        "- If your heuristic’s cost_ratio is well below the bound, it can safely become a bit greedier (larger estimates)\n"
        "  to further reduce generated nodes while still remaining valid.\n"
        "- Must be efficient: O(16) or better.\n"
        "- Avoid unnecessary nested loops.\n"
        "- Define all constants or lookup tables (like MANHATTAN_TABLE) **inside** the function.\n"
        "- Do NOT import, print, or reference any global variables or external files.\n"
        "\n"
        "Scoring metrics:\n"
        "- generated_ratio: lower is better (fewer unique nodes generated)\n"
        f"- cost_ratio: must be ≤ {COST_BOUND:.2f}\n"
        "\n"
        f"Current best: generated={high_meta['expanded_ratio']:.3f}, cost={high_meta['cost_ratio']:.3f}, score={high_score:.4f}\n"
        f"Worse example: generated={low_meta['expanded_ratio']:.3f}, cost={low_meta['cost_ratio']:.3f}, score={low_score:.4f}\n"
        "\n"
        "BE CREATIVE, BE CREATIVE, theres reward for being CREATIVE, attempt and try beyond existing and classic heuristics, always return **valid Python code** implementing:\n"
        "    def heuristic(s: State) -> int\n"
        "\n"
        "⚠️ **VERY IMPORTANT — OUTPUT FORMAT REQUIREMENTS:** ⚠️\n"
        "- Output ONLY one JSON object.\n"
        "- The JSON must contain a single key named \"code\".\n"
        "- The value must be a string containing your full Python code.\n"
        "- Do NOT include explanations, markdown, or triple backticks.\n"
        "- Do NOT add text before or after the JSON.\n"
        "\n"
        "✅ **Example of correct output format:**\n"
        "{\n"
        "  \"code\": \"from fifteen_state_class import State\\n\\n"
        "def heuristic(s: State) -> int:\\n"
        "    dist = 0\\n"
        "    for i, val in enumerate(s.tiles):\\n"
        "        if val == 0: continue\\n"
        "        goal_r, goal_c = divmod(val, 4)\\n"
        "        cur_r, cur_c = divmod(i, 4)\\n"
        "        dist += abs(goal_r - cur_r) + abs(goal_c - cur_c)\\n"
        "    return dist\"\n"
        "}\n"
        "\n"
        "🧩 **Now output ONLY your improved heuristic in that JSON format.**"
    )

    

    prompt = header

    # === Previous round heuristics ===
    if prev_list:
        for i, (pc, ps) in enumerate(prev_list):
            if pc:
                prev_def = try_extract_heuristic(pc)
                if prev_def:
                    prompt += f"\n# === Previous (round -{len(prev_list)-i}) [score={ps:.4f}] ===\n"
                    prompt += rename_func(prev_def, f'heuristic_prev{i}') + "\n"

    # === Low (worse) and High (better) examples ===
    if low_def is not None:
        prompt += (
            f"\n# === Low (worse) heuristic_v0 [score={low_score:.4f}, "
            f"exp={low_meta['expanded_ratio']:.3f}, cost={low_meta['cost_ratio']:.3f}] ===\n"
            + rename_func(low_def, 'heuristic_v0')
        )

    prompt += (
        f"\n\n# === High (better) heuristic_v1 [score={high_score:.4f}, "
        f"exp={high_meta['expanded_ratio']:.3f}, cost={high_meta['cost_ratio']:.3f}] ===\n"
        + rename_func(high_def, 'heuristic_v1')
    )

    # === Random heuristics for exploration ===
    if rand_list:
        for idx, (rc, rs) in enumerate(rand_list):
            rdef = try_extract_heuristic(rc)
            if rdef:
                prompt += f"\n\n# === Random heuristic_vr{idx} [score={rs:.4f}] ===\n{rename_func(rdef, f'heuristic_vr{idx}')}"

    prompt += "\n\n# === Now produce improved final `heuristic` ===\n"
    return prompt


# ====== IO ======
FOLDER = "generated_programs"
os.makedirs(FOLDER, exist_ok=True)

def write_code(island_id: int, cnt: int, code: str) -> str:
    path = os.path.join(FOLDER, f"generated_program_{island_id}_{cnt}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return path

def load_heuristic_from_file(path: str):
    unique_name = f"mod_{os.path.basename(path).replace('.py','')}_{int(time.time()*1e6)}"
    spec = importlib.util.spec_from_file_location(unique_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "heuristic")

# ====== Model call ======
client = None
API_MAX_CONCURRENCY = 8
api_sem = None

def model_generate(prompt: str) -> str:
    global client
    backoff = 0.6
    models = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview","gemini-3-pro-preview"]
    random.shuffle(models)
    last_err = None
    for i in range(5):
        model = models[min(i, len(models)-1)]
        # print("model used:")
        # print(model)
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                config={
                    "response_mime_type": "application/json",
                    "system_instruction": "Return ONLY JSON {\"code\": \"...\"}",
                },
            )
            code = validate_api_json(getattr(resp, "text", "") or "")
            if code:
                return code
            raise RuntimeError("invalid payload")
        except Exception as e:
            last_err = e
            print(f"[WARN] model error: {e}")
            time.sleep(backoff)
            backoff = min(8.0, backoff * 2)
    return "from fifteen_state_class import State\n\ndef heuristic(s: State) -> int:\n    # fallback heuristic: returns 0 for all states\n    return 0\n"

async def model_generate_async(prompt: str) -> str:
    async with api_sem:
        return await asyncio.to_thread(model_generate, prompt)

# ====== Evaluation ======
def _eval_worker(path: str, out_q: mp.Queue):
    global COST_BOUND
    try:
        import evaluate_max
        import sampler
        heuristic = load_heuristic_from_file(path)
        # Ensure the fixed train/test split exists (seeded randomness)
        if not (os.path.exists(TRAIN_FULL_FILE) and os.path.exists(TEST_FULL_FILE)):
            import sampler
            sampler.split_train_test(
                input_file="test_full.txt",
                train_file=TRAIN_FULL_FILE,
                test_file=TEST_FULL_FILE,
                train_size=15,
                seed=243,
            )

        generated_ratio, cost_ratio = evaluate_max.evaluate_astar(heuristic, TRAIN_FULL_FILE)
        is_valid = cost_ratio <= COST_BOUND
        # The "score" you optimize is now the generated_ratio (lower is better)
        score = generated_ratio

        out_q.put((generated_ratio, cost_ratio, score, is_valid))
    except Exception as e:
        out_q.put(e)


def evaluate_file(path: str):
    """
    Evaluate a heuristic file safely under PyPy.
    If evaluation fails (timeout/crash), return a finite penalty score
    but still mark is_valid = False.
    """
    BAD_SCORE = 999.0  # finite fallback for evaluation failures

    out_q = CTX.Queue()
    proc = CTX.Process(target=_eval_worker, args=(path, out_q))

    try:
        proc.start()
        proc.join(EVAL_TIMEOUT_SEC)

        # --- Handle timeout ---
        if proc.is_alive():
            proc.terminate()
            proc.join(1)
            print(f"[WARN] timeout evaluating {path}")
            return BAD_SCORE, BAD_SCORE, BAD_SCORE, False

        # --- Try to get result from queue ---
        try:
            result = out_q.get_nowait()
        except Exception as e:
            print(f"[WARN] eval infra error: {e}")
            return BAD_SCORE, BAD_SCORE, BAD_SCORE, False

        # --- Worker raised exception ---
        if isinstance(result, Exception):
            print(f"[WARN] eval failed: {result}")
            return BAD_SCORE, BAD_SCORE, BAD_SCORE, False

        # --- Normal result ---
        if isinstance(result, tuple) and len(result) == 4:
            generated_ratio, cost_ratio, score, is_valid = result
            status = "VALID" if is_valid else "HIGH_COST"
            print(f"[EVAL] generated={generated_ratio:.4f} cost={cost_ratio:.4f} -> {status}")
            return generated_ratio, cost_ratio, score, is_valid

        # --- Unexpected structure ---
        print(f"[EVAL] unexpected result: {result}")
        return BAD_SCORE, BAD_SCORE, BAD_SCORE, False

    except Exception as e:
        print(f"[FATAL] evaluate_file crashed: {e}")
        return BAD_SCORE, BAD_SCORE, BAD_SCORE, False

    finally:
        # ===== Safe cleanup =====
        import gc, time
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass

        try:
            proc.close()
        except Exception:
            pass

        try:
            out_q.close()
        except Exception:
            pass

        try:
            out_q.join_thread()
        except Exception:
            pass

        time.sleep(0.05)
        gc.collect()

async def evaluate_file_async(path: str) -> float:
    return await asyncio.to_thread(evaluate_file, path)

# ====== Selection ======
def select_low_high_rand_from(results_list: List[Tuple[int, float, str, float, float, bool]]):
    # Expected record: (cnt, score, path, expanded_ratio, cost_ratio, is_valid)

    # Keep only programs with actual file
    entries = [r for r in results_list if r[2] and os.path.exists(r[2])]
    if len(entries) < 2:
        raise RuntimeError("need at least two programs")

    # Split by validity
    valids   = [r for r in entries if r[5] is True]
    invalids = [r for r in entries if r[5] is False]

    # If no valids yet -> bootstrap: treat invalids as valids
    if not valids:
        valids = invalids.copy()

    # -------------------------------
    # HIGH and WORST must be VALID ONLY
    # -------------------------------
    sorted_valid = sorted(valids, key=lambda x: x[1])  # sort by score
    best  = sorted_valid[0]        # lowest score valid
    worst = sorted_valid[-1]       # highest score valid

    # -------------------------------
    # Random pool logic
    # -------------------------------
    # Key idea:
    # - If we have enough valid heuristics, use mostly valid random seeds
    # - If not, allow invalids to inject diversity
    # -------------------------------
    if len(valids) >= 4:
        # Mature phase → mostly use valid ones (except best/worst)
        rand_pool = sorted_valid[1:-1]  # middle valid heuristics
    else:
        # Early bootstrap → mix valid & invalid to explore
        rand_pool = valids + invalids

    # random pick up to 3 entries
    rand_choices = random.sample(rand_pool, min(3, len(rand_pool))) if rand_pool else []

    # Extract code + score for high/low/random
    low_code  = safe_read_file(worst[2])
    low_score = worst[1]
    low_meta  = {"expanded_ratio": worst[3], "cost_ratio": worst[4]}

    high_code  = safe_read_file(best[2])
    high_score = best[1]
    high_meta  = {"expanded_ratio": best[3], "cost_ratio": best[4]}

    rand_codes = [
        (safe_read_file(r[2]), r[1])
        for r in rand_choices
        if safe_read_file(r[2])
    ]

    return (low_code, low_score, low_meta), (high_code, high_score, high_meta), rand_codes

# ====== Iteration ======
async def generate_one_iteration_async(state: IslandState) -> Tuple[int, float]:
    try:
        # === 1️⃣ select best/worst + randoms ===
        (low_code, low_score, low_meta), (high_code, high_score, high_meta), rand_list = \
            select_low_high_rand_from(state.results)

        # 🔹 get recent two versions for context
        prev_list = []
        for i in range(1, 3):  # last two iterations
            if len(state.results) >= i:
                _, prev_score, prev_path, _, _, _ = state.results[-i]
                prev_code = safe_read_file(prev_path)
                if prev_code:
                    prev_list.insert(0, (prev_code, prev_score))

        # === 2️⃣ build prompt (now includes ratios) ===
        prompt = build_best_shot_prompt(
            low_code=low_code,
            low_score=low_score,
            low_meta=low_meta,
            high_code=high_code,
            high_score=high_score,
            high_meta=high_meta,
            rand_list=rand_list,
            prev_list=prev_list,
        )

        # === 3️⃣ generate + evaluate ===
        new_code = await model_generate_async(prompt)
        path = write_code(state.island_id, state.cnt, new_code)

        # evaluate synchronously (no weighted score)
        expanded_ratio, cost_ratio, score, is_valid = await asyncio.to_thread(evaluate_file, path)

    except Exception as e:
        print(f"[WARN] island {state.island_id} v{state.cnt} gen failed: {e}")
        path = os.path.join(FOLDER, f"failed_{state.island_id}_{state.cnt}.py")
        with open(path, "w") as f:
            f.write("# failed\n")
        BAD_SCORE = 999.0
        expanded_ratio, cost_ratio, score, is_valid = BAD_SCORE, BAD_SCORE, BAD_SCORE, False


    # === 4️⃣ record result ===
    state.results.append((state.cnt, score, path, expanded_ratio, cost_ratio, is_valid))
    print(
        f"[Island {state.island_id} | v{state.cnt}] "
        f"exp={expanded_ratio:.4f} cost={cost_ratio:.4f} valid={is_valid}"
    )
    state.cnt += 1
    return state.cnt - 1, score


# ====== Bootstrap ======
def bootstrap_islands(num_islands: int = 8) -> List[IslandState]:
    """
    Initialize all islands with two starting admissible programs:
    - generated_program_0.py: Manhattan
    - generated_program_1.py: Linear Conflict
    """
    init_meta = [
        # (ver, score, path, expanded_ratio, cost_ratio, is_valid)
        (0, 1.0000, os.path.join(FOLDER, "generated_program_0.py"), 1.0, 1.0, True),
        (1, 1.0000, os.path.join(FOLDER, "generated_program_1.py"), 1.0, 1.0, True),
    ]

    states = []
    for i in range(num_islands):
        st = IslandState(island_id=i)
        for ver, score, path, expanded, cost, valid in init_meta:
            code = safe_read_file(path)
            if not code:
                continue
            new_path = write_code(i, ver, code)
            st.results.append((ver, float(score), new_path, expanded, cost, valid))
            print(
                f"[Bootstrap] island={i} v={ver} score={score:.4f} "
                f"exp={expanded:.2f} cost={cost:.2f} valid={valid} -> {new_path}"
            )
        st.cnt = len(st.results)
        states.append(st)

    return states

# ====== Cull & refill ======
def cull_and_refill(states: List[IslandState], MIGRATION_RATE: int = 2):
    """
    Cull the worst half of islands and inject a small number of elite
    heuristics from survivors. We DO NOT wipe island history.
    """

    # -------------------------------------------------------
    # 1. Identify best individual from each island
    # -------------------------------------------------------
    island_bests = []
    for st in states:
        if st.results:
            valids = [r for r in st.results if r[5]]
            if valids:
                br = min(valids, key=lambda x: x[1])    # best valid
            else:
                br = min(st.results, key=lambda x: x[1]) # fallback
            island_bests.append((st, br))
        else:
            island_bests.append((st, (None, float("inf"), "", 1.0, 1.0, False)))

    # -------------------------------------------------------
    # 2. Rank islands by best score
    # -------------------------------------------------------
    ranked = sorted(island_bests, key=lambda x: x[1][1])  # sort by score
    survivors = [st for st, _ in ranked[:len(states)//2]]
    culled    = [st for st, _ in ranked[len(states)//2:]]

    print("\n=== CULL ===")
    print("Survivors:", [s.island_id for s in survivors])
    print("Culled:", [c.island_id for c in culled])

    # -------------------------------------------------------
    # 3. Collect top elites from survivors (valid-only preferred)
    # -------------------------------------------------------
    survivor_best_snippets = []
    for st in survivors:
        vr = [r for r in st.results if r[5]] or st.results
        br = min(vr, key=lambda x: x[1])
        cnt, score, path, exp_r, cost_r, valid = br[:6]
        survivor_best_snippets.append((score, safe_read_file(path), exp_r, cost_r, valid))

    # Sort survivor elites by best score
    survivor_best_snippets.sort(key=lambda x: x[0])  # lower score = better

    # -------------------------------------------------------
    # 4. Inject only top MIGRATION_RATE elites per culled island
    # -------------------------------------------------------
    for st in culled:
        print(f"[Inject] island {st.island_id}")

        # Take the top few elites (default = 2)
        elites_to_inject = survivor_best_snippets[:MIGRATION_RATE]

        for (score, code_text, exp_r, cost_r, valid) in elites_to_inject:
            if not code_text:
                continue

            new_path = write_code(st.island_id, st.cnt, code_text)
            st.results.append((st.cnt, score, new_path, exp_r, cost_r, valid))
            print(f"   Injected elite with score={score:.4f}, cost={cost_r:.3f}, valid={valid}")
            st.cnt += 1

# ====== Summary ======
def island_summary_str(state: IslandState) -> str:
    import math
    try:
        if not state.results:
            return f"Island {state.island_id} | empty"

        # split by validity
        valids = [r for r in state.results if len(r) >= 6 and r[5] is True]
        invalids = [r for r in state.results if len(r) >= 6 and r[5] is False]

        # last score for delta display
        last = state.results[-1]
        last_score = last[1] if len(last) >= 2 and isinstance(last[1], (int, float)) else float("inf")

        # mean cost over finite costs
        finite_costs = [
            r[4] for r in state.results
            if len(r) >= 6 and isinstance(r[4], (int, float)) and math.isfinite(r[4])
        ]
        mean_cost = (sum(finite_costs) / len(finite_costs)) if finite_costs else float("inf")

        valid_count = len(valids)
        invalid_count = len(invalids)
        valid_rate = valid_count / len(state.results) if state.results else 0.0

        if valids:
            best = min(valids, key=lambda x: x[1])  # minimize score among valid only
            best_score, best_expanded, best_cost = best[1], best[3], best[4]
            delta = last_score - best_score
            return (
                f"Island {state.island_id} | best={best_score:.4f} "
                f"(exp={best_expanded:.4f}, cost={best_cost:.4f}) "
                f"| last={last_score:.4f} | Δ={delta:+.4f} | n={len(state.results)} "
                f"| valid={valid_count} | invalid={invalid_count} | valid_rate={valid_rate:.2f} "
                f"| mean_cost={(mean_cost if math.isfinite(mean_cost) else float('inf')) if isinstance(mean_cost, float) else mean_cost}"
            )
        else:
            # no feasible heuristics this island
            return (
                f"Island {state.island_id} | best=inf (no valid heuristics) "
                f"| last={last_score:.4f} | Δ=N/A | n={len(state.results)} "
                f"| valid=0 | invalid={invalid_count} | valid_rate=0.00 "
                f"| mean_cost={(mean_cost if math.isfinite(mean_cost) else float('inf')) if isinstance(mean_cost, float) else mean_cost}"
            )

    except Exception as e:
        return f"Island {state.island_id} | summary_error: {e}"


def print_summary(states: List[IslandState], round_idx: int):
    print(f"\n===== ROUND {round_idx+1} SUMMARY =====")
    for st in states:
        print(island_summary_str(st))

    all_results = [
        (st.island_id, *rec)
        for st in states
        for rec in st.results if rec
    ]
    if not all_results:
        return

    # Only consider valid results for GLOBAL_BEST
    valid_results = [r for r in all_results if len(r) >= 7 and r[6] is True]

    if valid_results:
        gbest = min(valid_results, key=lambda x: x[2])  # x[2] = score
        print(
            f"GLOBAL_BEST: island={gbest[0]} v={gbest[1]} "
            f"score={gbest[2]:.4f} | exp={gbest[4]:.3f} cost={gbest[5]:.3f} | valid={gbest[6]}"
        )
    else:
        # Explicitly say “no valid heuristic under bound yet”
        print("GLOBAL_BEST: none (no valid heuristics under COST_BOUND yet)")

def get_global_best_valid(states: List[IslandState]):
    """
    Returns (island_id, rec) where rec is (cnt, score, path, expanded_ratio, cost_ratio, is_valid).
    Only considers valid records. Returns None if no valid record exists yet.
    """
    all_valid = []
    for st in states:
        for rec in st.results:
            if not rec or len(rec) < 6:
                continue
            if rec[5] is True:
                all_valid.append((st.island_id, rec))

    if not all_valid:
        return None

    # rec[1] is score
    return min(all_valid, key=lambda x: x[1][1])


def eval_heuristic_on_filepath(heuristic_fn, filepath: str) -> Tuple[float, float]:
    """
    Wrapper to make intent explicit: returns (avg_generated_ratio, max_cost_ratio).
    """
    import evaluate_max
    return evaluate_max.evaluate_astar(heuristic_fn, filepath)


def evaluate_best_fullsets(states: List[IslandState], round_idx: int) -> Optional[BestEvalRecord]:
    gbest = get_global_best_valid(states)
    if gbest is None:
        return None

    island_id, rec = gbest
    ver, score, path, _, _, _ = rec[:6]
    heuristic = load_heuristic_from_file(path)

    # always evaluate train
    gen_train, cost_train = eval_heuristic_on_filepath(heuristic, TRAIN_FULL_FILE)

    # conditionally evaluate test
    if round_idx % TEST_EVAL_INTERVAL == 0:
        gen_test, cost_test = eval_heuristic_on_filepath(heuristic, TEST_FULL_FILE)
    else:
        gen_test, cost_test = float("nan"), float("nan")

    ts = datetime.datetime.now().isoformat(timespec="seconds")
    return BestEvalRecord(
        round_idx=round_idx,
        wall_time_iso=ts,
        island_id=island_id,
        version=int(ver),
        score=float(score),
        gen_train=float(gen_train),
        cost_train=float(cost_train),
        gen_test=float(gen_test),
        cost_test=float(cost_test),
    )

def append_checkpoint_csv(records: List[BestEvalRecord], csv_path: str):
    # Writes full file each time to keep it simple and robust
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "round",
            "time",
            "island",
            "version",
            "score",
            "gen_train",
            "cost_train",
            "gen_test",
            "cost_test",
        ])
        for r in records:
            w.writerow([
                r.round_idx,
                r.wall_time_iso,
                r.island_id,
                r.version,
                f"{r.score:.6f}",
                f"{r.gen_train:.6f}",
                f"{r.cost_train:.6f}",
                f"{r.gen_test:.6f}",
                f"{r.cost_test:.6f}",
            ])


def print_final_table(records: List[BestEvalRecord]):
    if not records:
        print("\n=== BEST CHECKPOINT HISTORY ===")
        print("No valid heuristic was found under COST_BOUND, so no checkpoints to report.")
        return

    headers = ["round", "island", "v", "score", "gen_train", "cost_train", "gen_test", "cost_test"]
    rows = []
    for r in records:
        rows.append([
            r.round_idx,
            r.island_id,
            r.version,
            f"{r.score:.6f}",
            f"{r.gen_train:.6f}",
            f"{r.cost_train:.6f}",
            f"{r.gen_test:.6f}",
            f"{r.cost_test:.6f}",
        ])

    # Pretty print without extra deps
    colw = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            colw[i] = max(colw[i], len(str(cell)))

    def fmt_row(row):
        return " | ".join(str(row[i]).rjust(colw[i]) for i in range(len(headers)))

    print("\n=== BEST CHECKPOINT HISTORY ===")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in colw))
    for row in rows:
        print(fmt_row(row))


def plot_history(records: List[BestEvalRecord], out_png: str):
    if not records:
        return

    import matplotlib.pyplot as plt

    xs = [r.round_idx for r in records]
    gen_tr = [r.gen_train for r in records]
    gen_te = [r.gen_test for r in records]
    cost_tr = [r.cost_train for r in records]
    cost_te = [r.cost_test for r in records]

    # Plot 1: avg generated ratio
    plt.figure()
    plt.plot(xs, gen_tr, label="train avg_generated_ratio")
    plt.plot(xs, gen_te, label="test avg_generated_ratio")
    plt.xlabel("round")
    plt.ylabel("avg generated ratio")
    plt.title("Global best: avg generated ratio over rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_generated.png"))
    plt.close()

    # Plot 2: max cost ratio
    plt.figure()
    plt.plot(xs, cost_tr, label="train max_cost_ratio")
    plt.plot(xs, cost_te, label="test max_cost_ratio")
    plt.axhline(COST_BOUND, linestyle=":", label="COST_BOUND")
    plt.xlabel("round")
    plt.ylabel("max cost ratio")
    plt.title("Global best: max cost ratio over rounds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png.replace(".png", "_cost.png"))
    plt.close()


# ====== Helpers ======
MAX_CODE_CHARS = 200_000
EVAL_TIMEOUT_SEC = 1500
COST_BOUND = 1.75  # max allowed cost ratio
TEST_EVAL_INTERVAL = 1   # only run test_full every 4 rounds


# ====== Main ======
async def main_multi_islands():
    states = bootstrap_islands(num_islands=8)
    TOTAL_ROUNDS = 23
    CHECKPOINT_INTERVAL = 8
    SUMMARY_INTERVAL = 1
    
    for r in range(TOTAL_ROUNDS):
        tasks = [asyncio.create_task(generate_one_iteration_async(st)) for st in states]
        results_or_exc = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, res in enumerate(results_or_exc):
            if isinstance(res, Exception):
                print(f"[WARN] task failed on island {states[idx].island_id}: {res}")

        if (r + 1) % SUMMARY_INTERVAL == 0:
            print_summary(states, r)

        # New: evaluate global best valid on full train and full test each round
        try:
            rec = await asyncio.to_thread(evaluate_best_fullsets, states, r + 1)
            if rec is None:
                print("[CHECKPOINT] no valid heuristic under COST_BOUND yet")
            else:
                BEST_HISTORY.append(rec)
                append_checkpoint_csv(BEST_HISTORY, CHECKPOINT_CSV)

                if not math.isnan(rec.gen_test):
                    print(
                        f"[CHECKPOINT] round={rec.round_idx} best=island {rec.island_id} v{rec.version} "
                        f"train(gen={rec.gen_train:.4f}, cost={rec.cost_train:.4f}) "
                        f"test(gen={rec.gen_test:.4f}, cost={rec.cost_test:.4f})"
                    )
                else:
                    print(
                        f"[CHECKPOINT] round={rec.round_idx} best=island {rec.island_id} v{rec.version} "
                        f"train(gen={rec.gen_train:.4f}, cost={rec.cost_train:.4f}) "
                        f"test=SKIPPED"
                    )

        except Exception as e:
            print(f"[WARN] checkpoint eval failed: {e}")

        if (r + 1) % CHECKPOINT_INTERVAL == 0:
            cull_and_refill(states)
            print_summary(states, r)

    print("\n=== Final Results ===")
    for st in states:
        if st.results:
            best_valids = [r for r in st.results if len(r) >= 6 and r[5] is True]
            if best_valids:
                best = min(best_valids, key=lambda x: x[1])
                print(f"Island {st.island_id}: v{best[0]} score={best[1]:.4f} (valid, cost ≤ {COST_BOUND})")
            else:
                best = min(st.results, key=lambda x: x[1])
                print(
                    f"Island {st.island_id}: v{best[0]} score={best[1]:.4f} "
                    f"(NO VALID HEURISTIC, best cost={best[4]:.3f} > {COST_BOUND})"
                )

        # --------------------------------------
        # === ADD THIS BLOCK (global best) ===
        # --------------------------------------

    # compute global best VALID heuristic
    all_results = [(st.island_id, *rec)
                for st in states
                for rec in st.results
                if rec]

    valid_results = [r for r in all_results if len(r) >= 7 and r[6] is True]

    if valid_results:
        gbest = min(valid_results, key=lambda x: x[2]) # minimize score
        print(
            f"\nGLOBAL BEST: island={gbest[0]} v={gbest[1]} "
            f"score={gbest[2]:.4f} | exp={gbest[4]:.3f} "
            f"cost={gbest[5]:.3f} | valid={gbest[6]}"
        )
    else:
        print("\nGLOBAL BEST: none (no valid heuristics under COST_BOUND)")




if __name__ == "__main__":
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    # 只在主进程创建这些资源
    client = genai.Client(api_key="")
    API_MAX_CONCURRENCY = 8
    api_sem = asyncio.Semaphore(API_MAX_CONCURRENCY)

    log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, f"log_{timestamp}.txt")
    sys.stdout = Logger(logfile)
    asyncio.run(main_multi_islands())

    # ✅ Add this block at the end
    import gc, atexit

    @atexit.register
    def cleanup_resources():
        """Force garbage collection and release leaked semaphores (PyPy fix)."""
        import multiprocessing as mp
        try:
            # Touch active children to trigger cleanup of resource tracker
            mp.active_children()
        except Exception:
            pass
        gc.collect()
        print("[CLEANUP] Final GC cycle completed safely.")

    # Final reporting for best-over-time
    print_final_table(BEST_HISTORY)
    try:
        plot_history(BEST_HISTORY, CHECKPOINT_PNG)
        print(f"[PLOT] saved {CHECKPOINT_PNG.replace('.png','_generated.png')} and {CHECKPOINT_PNG.replace('.png','_cost.png')}")
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")

    print(f"[CSV] saved {CHECKPOINT_CSV}")
