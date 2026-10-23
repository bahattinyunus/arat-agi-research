from typing import Dict, List, Set, Tuple, Optional
from collections import deque

class CausalDAG:
    """
    Directed Acyclic Graph (DAG) for Causal Modeling and d-Separation analysis.
    Implements Judea Pearl's d-separation criterion to evaluate conditional independence:
    (X _||_ Y | Z) in G <=> X and Y are d-separated by Z.
    """
    def __init__(self):
        self.nodes: Set[str] = set()
        self.parents: Dict[str, Set[str]] = {}
        self.children: Dict[str, Set[str]] = {}

    def add_node(self, node: str):
        if node not in self.nodes:
            self.nodes.add(node)
            self.parents[node] = set()
            self.children[node] = set()

    def add_edge(self, u: str, v: str):
        """Adds directed causal edge u -> v (u causes v)."""
        self.add_node(u)
        self.add_node(v)
        self.children[u].add(v)
        self.parents[v].add(u)

    def get_ancestors(self, node: str) -> Set[str]:
        """Returns all causal ancestors of a node including itself."""
        ancestors = set()
        queue = deque([node])
        while queue:
            curr = queue.popleft()
            if curr not in ancestors:
                ancestors.add(curr)
                queue.extend(self.parents.get(curr, set()))
        return ancestors

    def topological_sort(self) -> List[str]:
        """Returns topological ordering of nodes for causal generation."""
        in_degree = {n: len(self.parents[n]) for n in self.nodes}
        queue = deque([n for n, deg in in_degree.items() if deg == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for child in self.children[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise ValueError("Cycle detected in Causal DAG!")
        return order

    def is_d_separated(self, x: str, y: str, conditioning_set: Set[str]) -> bool:
        """
        Evaluates whether nodes x and y are d-separated given conditioning_set Z.
        A path is blocked by Z if:
        1. It contains a chain (A -> B -> C) or fork (A <- B -> C) where B is in Z.
        2. It contains a collider (A -> B <- C) where neither B nor any descendant of B is in Z.
        """
        if x == y:
            return False

        # Gather all ancestors of conditioning set for collider check
        conditioning_ancestors = set(conditioning_set)
        for z in conditioning_set:
            conditioning_ancestors.update(self.get_ancestors(z))

        # Find all undirected paths between x and y
        def find_paths(current: str, target: str, visited: Set[str]) -> List[List[str]]:
            if current == target:
                return [[target]]
            paths = []
            visited.add(current)
            neighbors = self.parents.get(current, set()) | self.children.get(current, set())
            for nbr in neighbors:
                if nbr not in visited:
                    sub_paths = find_paths(nbr, target, set(visited))
                    for sp in sub_paths:
                        paths.append([current] + sp)
            return paths

        all_paths = find_paths(x, y, set())
        if not all_paths:
            return True

        # Check if every path between x and y is blocked
        for path in all_paths:
            path_blocked = False
            for i in range(1, len(path) - 1):
                prev_n = path[i - 1]
                mid_n = path[i]
                next_n = path[i + 1]

                is_collider = (prev_n in self.parents.get(mid_n, set())) and (next_n in self.parents.get(mid_n, set()))

                if is_collider:
                    # Collider: blocked if mid_n and none of its descendants are in Z
                    if mid_n not in conditioning_ancestors:
                        path_blocked = True
                        break
                else:
                    # Chain or Fork: blocked if mid_n is in Z
                    if mid_n in conditioning_set:
                        path_blocked = True
                        break

            if not path_blocked:
                # Found an unblocked (active) path
                return False

        return True
