import pytest
from core_cognition.constructor_theory import (
    PhysicalState,
    PhysicalTask,
    Constructor,
    ConstructorTaskPlanner
)

def test_physical_state_matching_and_cloning():
    s1 = PhysicalState("LowEnergySubstrate", {"temp": 25, "charge": 0.2}, energy=20.0, entropy=1.5)
    s2 = PhysicalState("LowEnergySubstrate", {"temp": 25, "charge": 0.2}, energy=20.0, entropy=1.5)
    s3 = PhysicalState("HighEnergySubstrate", {"temp": 50, "charge": 0.9}, energy=90.0, entropy=2.0)

    assert s1.matches(s2)
    assert not s1.matches(s3)

    clone = s1.clone()
    assert clone.matches(s1)
    clone.energy = 30.0
    assert not clone.matches(s1)

def test_thermodynamic_admissibility_and_work():
    s_in = PhysicalState("RawSubstrate", energy=10.0, entropy=2.0)
    s_out = PhysicalState("OrderedSubstrate", energy=40.0, entropy=1.0) # Requires delta_e=30 work

    # Task without enough work: Impossible task in Constructor Theory
    impossible_task = PhysicalTask("ViolatingTask", s_in, s_out, required_work=10.0)
    assert not impossible_task.is_thermodynamically_admissible()

    # Task with sufficient work: Admissible task
    admissible_task = PhysicalTask("ValidTask", s_in, s_out, required_work=35.0)
    assert admissible_task.is_thermodynamically_admissible()

def test_sequential_task_composition():
    s0 = PhysicalState("S0", energy=10.0, entropy=1.0)
    s1 = PhysicalState("S1", energy=20.0, entropy=1.0)
    s2 = PhysicalState("S2", energy=30.0, entropy=1.0)

    t1 = PhysicalTask("T1", s0, s1, required_work=15.0)
    t2 = PhysicalTask("T2", s1, s2, required_work=15.0)

    composite = t1.compose_sequential(t2)
    assert composite is not None
    assert composite.input_state.matches(s0)
    assert composite.output_state.matches(s2)
    assert composite.required_work == 30.0

    # Incompatible composition
    s_other = PhysicalState("Other", energy=5.0, entropy=0.5)
    t3 = PhysicalTask("T3", s_other, s2, required_work=5.0)
    assert t1.compose_sequential(t3) is None

def test_constructor_execution_and_catalytic_invariance():
    s_in = PhysicalState("ColdBattery", {"charged": False}, energy=10.0, entropy=2.0)
    s_out = PhysicalState("ChargedBattery", {"charged": True}, energy=50.0, entropy=1.5)
    task = PhysicalTask("RechargeBattery", s_in, s_out, required_work=45.0)

    charger = Constructor("PowerStation", max_power=100.0, degradation_rate=0.02)
    charger.register_capability("RechargeBattery")

    assert charger.can_perform(task)

    # Execute task
    substrate = s_in.clone()
    result = charger.execute_task(task, substrate)
    assert result is not None
    assert result.matches(s_out)
    assert charger.operating_cycles == 1
    assert charger.health < 1.0

    # Restore constructor
    charger.restore_cycle(maintenance_energy=10.0)
    assert charger.health == 1.0

def test_constructor_task_planner_verification():
    c1 = Constructor("ArmRobot", max_power=50.0)
    c1.register_capability("AssemblePart")

    c2 = Constructor("WelderRobot", max_power=80.0)
    c2.register_capability("WeldJoint")

    planner = ConstructorTaskPlanner([c1, c2])

    s_a = PhysicalState("RawA", energy=5.0, entropy=1.0)
    s_b = PhysicalState("PartB", energy=15.0, entropy=0.8)
    s_c = PhysicalState("FinishedC", energy=35.0, entropy=0.6)

    t1 = PhysicalTask("AssemblePart", s_a, s_b, required_work=20.0)
    t2 = PhysicalTask("WeldJoint", s_b, s_c, required_work=30.0)

    plan_res = planner.verify_plan([t1, t2])
    assert plan_res["all_performable"] is True
    assert plan_res["total_tasks"] == 2

    # Task requiring missing constructor
    t_missing = PhysicalTask("PaintChassis", s_b, s_c, required_work=10.0)
    plan_invalid = planner.verify_plan([t1, t_missing])
    assert plan_invalid["all_performable"] is False
