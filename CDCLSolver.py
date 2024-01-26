class CDCLSolver:
    def __init__(self, num_vars, clauses):
        self.num_vars = num_vars
        self.clauses = clauses
        self.assignment = [None] * (num_vars + 1)
        self.decision_stack = []
        self.learned_clauses = []

    def solve(self):
        while True:
            conflict_clause = self.unit_propagation()
            if conflict_clause is not None:
                if not self.resolve_conflict(conflict_clause):
                    return "UNSAT"
            elif all(self.assignment[1:]):
                return "SAT"
            else:
                var_to_assign = self.choose_variable()
                self.assign_variable(var_to_assign)

    def unit_propagation(self):
        while True:
            unit_clause = self.find_unit_clause()
            if unit_clause is not None:
                self.propagate_unit_assignment(unit_clause)
            else:
                return None

    def find_unit_clause(self):
        for clause in self.clauses:
            unassigned_literals = [lit for lit in clause if self.assignment[abs(lit)] is None]
            if len(unassigned_literals) == 1:
                return clause
        return None

    def propagate_unit_assignment(self, unit_clause):
        lit = [lit for lit in unit_clause if self.assignment[abs(lit)] is None][0]
        self.assignment[abs(lit)] = lit
        self.decision_stack.append(lit)

    def choose_variable(self):
        for var in range(1, self.num_vars + 1):
            if self.assignment[var] is None:
                return var

    def assign_variable(self, var):
        self.assignment[var] = var
        self.decision_stack.append(var)

    def resolve_conflict(self, conflict_clause):
        if len(self.decision_stack) == 0:
            return False  # UNSAT

        conflict_var = abs(conflict_clause[0])
        conflict_level = self.get_decision_level(conflict_var)

        self.learned_clauses.append(conflict_clause)
        while len(self.decision_stack) > 0 and self.get_decision_level(self.decision_stack[-1]) > conflict_level:
            self.backtrack()

        return True  # SAT

    def backtrack(self):
        last_decision = self.decision_stack.pop()
        self.assignment[abs(last_decision)] = None

    def get_decision_level(self, var):
        return len([lit for lit in self.decision_stack if abs(lit) == var])

def read_dimacs_cnf(filename):
    clauses = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("c"):
                continue
            if line.startswith("p cnf"):
                num_vars, num_clauses = map(int, line.strip().split()[2:])
            else:
                clause = list(map(int, line.strip().split()[:-1]))
                clauses.append(clause)
    return num_vars, clauses

filename = "DIMACS_files/turbo_easy/example_2.cnf"
num_vars, clauses = read_dimacs_cnf(filename)

solver = CDCLSolver(num_vars, clauses)
result = solver.solve()

print(result)
