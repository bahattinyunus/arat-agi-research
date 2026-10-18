import math
from typing import Dict, List, Set, Any, Optional

class PhysicalState:
    """
    Represents an input or output substrate state in Constructor Theory.
    Characterized by state attributes, internal energy, entropy, and structural integrity.
    """
    def __init__(self, name: str, attributes: Optional[Dict[str, Any]] = None, energy: float = 100.0, entropy: float = 1.0):
        self.name = name
        self.attributes = attributes or {}
        self.energy = float(energy)
        self.entropy = float(entropy)

    def matches(self, other: 'PhysicalState', tolerance: float = 1e-4) -> bool:
        """Determines if two states match in attributes and physical properties."""
        if self.name != other.name:
            return False
        if abs(self.energy - other.energy) > tolerance or abs(self.entropy - other.entropy) > tolerance:
            return False
        for k, v in self.attributes.items():
            if k not in other.attributes or other.attributes[k] != v:
                return False
        return True

    def clone(self) -> 'PhysicalState':
        return PhysicalState(
            name=self.name,
            attributes=dict(self.attributes),
            energy=self.energy,
            entropy=self.entropy
        )

    def __repr__(self) -> str:
        return f"<State '{self.name}' E={self.energy:.1f} S={self.entropy:.2f}>"


class PhysicalTask:
    """
    Represents a task T in Constructor Theory: T = { S_in -> S_out }.
    Specifies a physical transformation on substrates.
    """
    def __init__(self, name: str, input_state: PhysicalState, output_state: PhysicalState, required_work: float = 5.0):
        self.name = name
        self.input_state = input_state
        self.output_state = output_state
        self.required_work = required_work

    def is_thermodynamically_admissible(self) -> bool:
        """
        Constructor Theory Principle: A task is impossible if it violates
        thermodynamic conservation laws (e.g. creating energy from nothing, or overall entropy decrease without work).
        """
        delta_e = self.output_state.energy - self.input_state.energy
        # Energy deficit must be covered by work
        if delta_e > self.required_work + 1e-6:
            return False
        # Second law: If entropy decreases, work must be invested
        delta_s = self.output_state.entropy - self.input_state.entropy
        if delta_s < 0 and self.required_work <= 0:
            return False
        return True

    def compose_sequential(self, other: 'PhysicalTask', new_name: Optional[str] = None) -> Optional['PhysicalTask']:
        """
        Sequential composition T1 . T2: T1 output state must match T2 input state.
        """
        if not self.output_state.matches(other.input_state):
            return None
        composite_name = new_name or f"({self.name} >> {other.name})"
        total_work = self.required_work + other.required_work
        return PhysicalTask(
            name=composite_name,
            input_state=self.input_state.clone(),
            output_state=other.output_state.clone(),
            required_work=total_work
        )

    def __repr__(self) -> str:
        return f"Task({self.name}: {self.input_state.name} -> {self.output_state.name})"


class Constructor:
    """
    A Constructor C is an entity that can cause a task S_in -> S_out
    and retains the ability to do so again (catalytic invariance / self-restoration).
    """
    def __init__(self, name: str, max_power: float = 50.0, degradation_rate: float = 0.01):
        self.name = name
        self.max_power = max_power
        self.degradation_rate = degradation_rate
        self.operating_cycles = 0
        self.health = 1.0  # 1.0 = pristine, 0.0 = exhausted
        self.supported_task_names: Set[str] = set()

    def register_capability(self, task_name: str):
        self.supported_task_names.add(task_name)

    def can_perform(self, task: PhysicalTask) -> bool:
        """Checks if constructor can perform the task based on capability, power, and health."""
        if task.name not in self.supported_task_names and not any(cap in task.name for cap in self.supported_task_names):
            return False
        if task.required_work > self.max_power:
            return False
        if self.health < 0.1:
            return False
        if not task.is_thermodynamically_admissible():
            return False
        return True

    def execute_task(self, task: PhysicalTask, substrate: PhysicalState) -> Optional[PhysicalState]:
        """
        Executes the task on substrate. Verifies substrate matches task input,
        transforms it to output state, and restores constructor integrity (with small degradation).
        """
        if not self.can_perform(task):
            return None
        if not substrate.matches(task.input_state):
            return None

        # Apply transformation
        resulting_substrate = task.output_state.clone()
        
        # Constructor self-maintenance cycle
        self.operating_cycles += 1
        self.health = max(0.0, self.health - self.degradation_rate)

        return resulting_substrate

    def restore_cycle(self, maintenance_energy: float = 10.0):
        """Restores constructor health to 1.0 (maintaining constructor definition)."""
        self.health = min(1.0, self.health + 0.1 * maintenance_energy)


class ConstructorTaskPlanner:
    """
    Plans and verifies sequences of physical tasks within Constructor Theory.
    Distinguishes performable vs impossible transformations before robotic action execution.
    """
    def __init__(self, constructors: List[Constructor]):
        self.constructors = {c.name: c for c in constructors}

    def verify_plan(self, tasks: List[PhysicalTask]) -> Dict[str, Any]:
        """
        Verifies if an entire sequence of tasks is performable by available constructors.
        """
        valid = True
        step_reports = []

        for task in tasks:
            if not task.is_thermodynamically_admissible():
                valid = False
                step_reports.append({
                    "task": task.name,
                    "performable": False,
                    "reason": "Thermodynamically inadmissible"
                })
                continue

            # Find matching constructor
            candidate = next((c for c in self.constructors.values() if c.can_perform(task)), None)
            if candidate is None:
                valid = False
                step_reports.append({
                    "task": task.name,
                    "performable": False,
                    "reason": "No capable constructor found"
                })
            else:
                step_reports.append({
                    "task": task.name,
                    "performable": True,
                    "constructor": candidate.name,
                    "required_work": task.required_work
                })

        return {
            "all_performable": valid,
            "total_tasks": len(tasks),
            "step_reports": step_reports
        }
