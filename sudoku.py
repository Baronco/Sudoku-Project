from pulp import LpProblem,LpVariable,lpSum
from dataclasses import dataclass
from numpy import array


@dataclass
class sudoku:
    '''
    The Sudoku grid has 9x9 cells.
    This model has 81 decision variables
    Binary variables are used to solve this LP model.
    In total, the model has 9x9x9 = 729 decision variables.
    '''
    ROWS : 'typing.Any' = range(1, 10)
    COLS : 'typing.Any' = range(1, 10)
    VALS : 'typing.Any' = range(1, 10)

    '''
    Boxes are used for Define constrains
    '''
    Boxes: 'typing.Any' = None

    #choices saves the decision variables with its positions
    # choices : 'typing.Any' = None

    #Saves decision variables in a list
    result_vars : 'typing.Any' = None

    #initial valus from sudoku to solve
    inputs : 'typing.Any' = None

    def sudoku_solve(self, inputs):
        #instantiate ´problem class

        #initial valus from sudoku to solve
        self.inputs = inputs

        prob = LpProblem("Sudoku Problem")

        # The decision variables are created
        choices = LpVariable.dicts("Choice", 
                                   (self.VALS, self.ROWS, self.COLS), 
                                    cat='Binary')

        self.Boxes = [[(3 * i + k + 1, 3 * j + l + 1) 
                            for k in range(3) 
                            for l in range(3)]
                            for i in range(3)
                            for j in range(3)]

        #Constrain 1: for each cell to choose a single number from 1 to 9
        for i in self.ROWS:
            for j in self.COLS:
                prob += lpSum([choices[v][i][j] for v in self.VALS]) == 1

        #Constrain 2: for each row and columns to have unique number from 1 to 9
        for v in self.VALS:
            for r in self.ROWS:
                prob += lpSum([choices[v][r][c] for c in self.COLS]) == 1

            for c in self.COLS:
                prob += lpSum([choices[v][r][c] for r in self.ROWS]) == 1

            for b in self.Boxes:
                prob += lpSum([choices[v][r][c] for (r, c) in b]) == 1

        #Constrain 3: initial valus from sudoku to solve
        for (v, r, c) in self.inputs:
            prob += choices[v][r][c] == 1
        #Problem solver
        prob.solve()

        #Get the desicion variables
        self.result_vars = array([[None]*9]*9)
        
        for i in self.ROWS:
            aux_print = ''
            for j in self.COLS:
                for v in self.VALS:
                    if choices[v][i][j].varValue == 1:
                        if j == 3 or j == 6:
                            aux_print = aux_print + ' ' + str(v) + ' |'
                        else: 
                            aux_print = aux_print + ' ' + str(v)
                        self.result_vars[i-1][j-1] = v
                    
            print(f'{aux_print}')
            if i == 3 or i == 6:
                aux = '-'*len(aux_print)
                print(f'{aux}') 
          

