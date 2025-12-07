# brain.py
from array_backend import xp
import random
from dataclasses import dataclass
from typing import List, Tuple, Callable, Dict

# =========================
#  OPERATORS BIT-LEVEL
# =========================

# Tutti gli operatori lavorano su vettori 1D di bit {0,1}, dtype uint8 o bool.
BitArray = xp.ndarray  # shape (N,), dtype=uint8 o bool

def op_identity(x: BitArray) -> BitArray:
    return x

def op_not(x: BitArray) -> BitArray:
    return 1 - x

def op_shift_left_xor(x: BitArray) -> BitArray:
    """XOR tra x e x shiftato a sinistra (tipo mini LFSR)."""
    shifted = xp.roll(x, -1)
    return xp.bitwise_xor(x, shifted)

def op_shift_right_xor(x: BitArray) -> BitArray:
    shifted = xp.roll(x, 1)
    return xp.bitwise_xor(x, shifted)

def op_and_neighbors(x: BitArray) -> BitArray:
    """AND locale con vicini (filtro tipo convolution min)."""
    left = xp.roll(x, 1)
    right = xp.roll(x, -1)
    return xp.bitwise_and(x, xp.bitwise_and(left, right))

def op_majority_3(x: BitArray) -> BitArray:
    """Majority gate su finestre di 3 bit (left, self, right)."""
    left = xp.roll(x, 1)
    right = xp.roll(x, -1)
    s = left + x + right  # 0..3
    return (s >= 2).astype(xp.uint8)

OP_LIBRARY: Dict[str, Callable[[BitArray], BitArray]] = {
    "id": op_identity,
    "not": op_not,
    "sx_xor": op_shift_left_xor,
    "dx_xor": op_shift_right_xor,
    "and3": op_and_neighbors,
    "maj3": op_majority_3,
}
OP_NAMES: List[str] = list(OP_LIBRARY.keys())


# =========================
#  NODI & MAPPE
# =========================

@dataclass
class Node:
    """Nodo del lattice: tiene solo il nome dell'operatore."""
    op_name: str


class LatticeMap:
    """
    Lattice D-dimensionale di nodi.
    La geometria è fissa, ma ogni nodo può cambiare operatore.
    """
    def __init__(self, size: int, dim: int):
        """
        size: numero di nodi per dimensione (N)
        dim:  dimensione D
        N_nodi = size ** dim
        """
        self.size = size
        self.dim = dim
        self.nodes: List[Node] = [
            Node(op_name=random.choice(OP_NAMES))
            for _ in range(size ** dim)
        ]

    # --- mapping coordinate <-> indice lineare ---
    def coord_to_index(self, coord: Tuple[int, ...]) -> int:
        assert len(coord) == self.dim
        idx = 0
        base = 1
        for c in reversed(coord):
            idx += c * base
            base *= self.size
        return idx

    def index_to_coord(self, idx: int) -> Tuple[int, ...]:
        coord = []
        for _ in range(self.dim):
            coord.append(idx % self.size)
            idx //= self.size
        return tuple(reversed(coord))

    # --- accesso nodi ---
    def get_node(self, coord: Tuple[int, ...]) -> Node:
        return self.nodes[self.coord_to_index(coord)]

    def set_node_op(self, coord: Tuple[int, ...], op_name: str):
        assert op_name in OP_LIBRARY
        self.nodes[self.coord_to_index(coord)].op_name = op_name

    # --- vicini topologici (toroidale) ---
    def neighbors(self, coord: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        neigh = []
        for d in range(self.dim):
            for delta in (-1, 1):
                c = list(coord)
                c[d] = (c[d] + delta) % self.size
                neigh.append(tuple(c))
        return neigh

    # --- mutazione semplice ---
    def mutate(self, p_flip: float = 0.01):
        """Cambia l'operatore di alcuni nodi random."""
        for node in self.nodes:
            if random.random() < p_flip:
                node.op_name = random.choice(OP_NAMES)

    # --- clonazione ---
    def clone(self) -> "LatticeMap":
        new = LatticeMap(self.size, self.dim)
        new.nodes = [Node(op_name=n.op_name) for n in self.nodes]
        return new


# =========================
#  PROPAGATION ENGINE
# =========================

@dataclass
class BeamState:
    coord: Tuple[int, ...]
    state: BitArray     # vettore di bit
    steps: int


def apply_node(map_: LatticeMap, coord: Tuple[int, ...], state: BitArray) -> BitArray:
    node = map_.get_node(coord)
    op = OP_LIBRARY[node.op_name]
    return op(state)


def propagate(
    map_: LatticeMap,
    x0: BitArray,
    start_coord: Tuple[int, ...],
    max_steps: int = 16,
    beam_width: int = 4,
) -> List[BeamState]:
    """
    Propagatore generico:
    - parte da start_coord con stato bit x0
    - ad ogni step:
      * applica operatore nel nodo
      * genera rami verso i vicini
      * tiene al massimo beam_width traiettorie (beam search grezzo)
    Il criterio di selezione dei beam lo decide EXECUTOR (via scoring esterno),
    quindi qui manteniamo tutti i beam e ci limitiamo a tagliare per numero.
    """
    # stato iniziale
    x0 = x0.astype(xp.uint8)
    beams: List[BeamState] = [BeamState(coord=start_coord, state=x0, steps=0)]

    for _ in range(max_steps):
        new_beams: List[BeamState] = []
        for b in beams:
            new_state = apply_node(map_, b.coord, b.state)
            new_steps = b.steps + 1
            for nb in map_.neighbors(b.coord):
                new_beams.append(
                    BeamState(coord=nb, state=new_state.copy(), steps=new_steps)
                )

        if not new_beams:
            break

        # qui tagliamo SOLO per numero; l'ordinamento lo può gestire chi chiama
        random.shuffle(new_beams)
        beams = new_beams[:beam_width]

    return beams
