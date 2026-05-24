import random
from dataclasses import dataclass

LOW, HIGH = 5.0, 15.0
BITS = 16


def f(x: float) -> float:
    return 5 * x * x + 2 * x - 10


def decode(bits: str) -> float:
    max_num = (1 << BITS) - 1
    num = int(bits, 2)
    return LOW + (HIGH - LOW) * num / max_num


def encode(x: float) -> str:
    max_num = (1 << BITS) - 1
    scaled = round((x - LOW) / (HIGH - LOW) * max_num)
    scaled = max(0, min(max_num, scaled))
    return format(scaled, f"0{BITS}b")


def fitness(bits: str) -> float:
    # Минимизируем f(x), поэтому пригодность обратная.
    return 1.0 / (1.0 + f(decode(bits)))


def random_chromosome() -> str:
    return "".join(random.choice("01") for _ in range(BITS))


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def init_blanket(size: int) -> list[str]:
    if size == 1:
        return [encode((LOW + HIGH) / 2)]
    step = (HIGH - LOW) / (size - 1)
    return [encode(LOW + i * step) for i in range(size)]


def init_shotgun(size: int) -> list[str]:
    return [random_chromosome() for _ in range(size)]


def choose_parents(population: list[str], mode: str) -> tuple[str, str]:
    if mode == "A":  # случайная
        return tuple(random.sample(population, 2))

    first = random.choice(population)  # инбридинг
    second = min((p for p in population if p != first), key=lambda p: hamming(first, p), default=random.choice(population))
    return first, second


def cross_one_point(p1: str, p2: str) -> tuple[str, str]:
    point = random.randint(1, BITS - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def cross_ordered_one_point(p1: str, p2: str) -> tuple[str, str]:
    point = random.randint(1, BITS - 1)
    c1 = p1[:point] + "".join(sorted(p2[point:]))
    c2 = p2[:point] + "".join(sorted(p1[point:]))
    return c1, c2


def cross_cyclic(p1: str, p2: str) -> tuple[str, str]:
    # Обмениваем каждый второй ген начиная со случайной позиции.
    start = random.randint(0, BITS - 1)
    c1, c2 = list(p1), list(p2)
    for i in range(start, BITS, 2):
        c1[i], c2[i] = c2[i], c1[i]
    return "".join(c1), "".join(c2)


def cross_golden(p1: str, p2: str) -> tuple[str, str]:
    point = max(1, min(BITS - 1, int((BITS - 1) * 0.618)))
    return p1[:point] + p2[point:], p2[:point] + p1[point:]


def apply_crossover(p1: str, p2: str, mode: str, probability: float) -> tuple[str, str]:
    if random.random() > probability:
        return p1, p2
    if mode == "A":
        return cross_one_point(p1, p2)
    if mode == "E":
        return cross_ordered_one_point(p1, p2)
    if mode == "I":
        return cross_cyclic(p1, p2)
    return cross_golden(p1, p2)


def mutate_simple(bits: str) -> str:
    i = random.randrange(BITS)
    flipped = "1" if bits[i] == "0" else "0"
    return bits[:i] + flipped + bits[i + 1 :]


def mutate_golden_swap(bits: str) -> str:
    i = int((BITS - 1) * 0.382)
    j = int((BITS - 1) * 0.618)
    arr = list(bits)
    arr[i], arr[j] = arr[j], arr[i]
    return "".join(arr)


def apply_mutation(bits: str, mode: str, probability: float) -> str:
    if random.random() > probability:
        return bits
    if mode == "A":
        return mutate_simple(bits)
    return mutate_golden_swap(bits)


def roulette_pick(population: list[str], weights: list[float]) -> str:
    total = sum(weights)
    if total <= 0:
        return random.choice(population)
    r = random.uniform(0, total)
    s = 0.0
    for item, w in zip(population, weights):
        s += w
        if s >= r:
            return item
    return population[-1]


def proportional_selection(population: list[str], size: int) -> list[str]:
    weights = [fitness(ch) for ch in population]
    return [roulette_pick(population, weights) for _ in range(size)]


def best(population: list[str]) -> tuple[str, float, float]:
    ch = min(population, key=lambda b: f(decode(b)))
    x = decode(ch)
    return ch, x, f(x)


@dataclass
class Config:
    population_size: int = 10
    generations: int = 50
    crossover_probability: float = 0.70
    mutation_probability: float = 0.20
    init_mode: str = "A"
    parent_mode: str = "A"
    crossover_mode: str = "A"
    mutation_mode: str = "A"


def ask_int(prompt: str, default: int, min_value: int | None = None) -> int:
    raw = input(f"{prompt} [по умолчанию: {default}]: ").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            print("Некорректный ввод, беру значение по умолчанию.")
            value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


def ask_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [по умолчанию: {default}]: ").strip()
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            print("Некорректный ввод, беру значение по умолчанию.")
            value = default
    return min(1.0, max(0.0, value))


def ask_choice(prompt: str, options: list[str], default: str) -> str:
    raw = input(f"{prompt} {options} [по умолчанию: {default}]: ").strip().upper()
    if not raw:
        return default
    if raw in options:
        return raw
    print("Некорректный выбор, беру значение по умолчанию.")
    return default


def read_config() -> Config:
    print("Генетический алгоритм: min f(x) = 5x^2 + 2x - 10, x in [5, 15]")
    cfg = Config()
    cfg.population_size = ask_int("Размер начальной популяции", 10, min_value=2)
    cfg.generations = ask_int("Число генераций (не менее 50)", 50, min_value=50)
    cfg.crossover_probability = ask_float("Вероятность кроссинговера (0..1)", 1)
    cfg.mutation_probability = ask_float("Вероятность мутации (0..1)", 1)
    cfg.init_mode = ask_choice("Начальная популяция: A=одеяло, B=дробовик", ["A", "B"], "A")
    cfg.parent_mode = ask_choice("Селекция родителей: A=случайная, E=инбридинг", ["A", "E"], "A")
    cfg.crossover_mode = ask_choice("Кроссинговер: A, E, I, L", ["A", "E", "I", "L"], "A")
    cfg.mutation_mode = ask_choice("Мутация: A=простая, D=золотое сечение", ["A", "D"], "A")
    return cfg


def run_ga(cfg: Config) -> tuple[str, float, float]:
    population = init_blanket(cfg.population_size) if cfg.init_mode == "A" else init_shotgun(cfg.population_size)
    best_ch, best_x, best_f = best(population)

    for gen in range(1, cfg.generations + 1):
        p1, p2 = choose_parents(population, cfg.parent_mode)
        c1, c2 = apply_crossover(p1, p2, cfg.crossover_mode, cfg.crossover_probability)
        c1 = apply_mutation(c1, cfg.mutation_mode, cfg.mutation_probability)
        c2 = apply_mutation(c2, cfg.mutation_mode, cfg.mutation_probability)

        # Микроэволюция: добавляем немного потомков и снова отбираем фиксированный размер.
        population.extend([c1, c2])
        population = proportional_selection(population, cfg.population_size)

        cur_ch, cur_x, cur_f = best(population)
        if cur_f < best_f:
            best_ch, best_x, best_f = cur_ch, cur_x, cur_f

        if gen == 1 or gen % 10 == 0 or gen == cfg.generations:
            print(f"Поколение {gen:3d}: x = {cur_x:.6f}, f(x) = {cur_f:.6f}")

    return best_ch, best_x, best_f


def main() -> None:
    random.seed()
    cfg = read_config()
    print("\nЗапуск...\n")
    ch, x, fx = run_ga(cfg)

    print("\nЛучший найденный результат:")
    print(f"Хромосома: {ch}")
    print(f"x = {x:.6f}")
    print(f"f(x) = {fx:.6f}")
    print(f"Теоретически на отрезке минимум в x = {LOW:.6f}, f(x) = {f(LOW):.6f}")


if __name__ == "__main__":
    main()
