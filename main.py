from sudoku import *
from pandas import read_excel, DataFrame


def sudoku_inputs():
    sudoku_inputs = read_excel(io='inputs.xlsm',sheet_name='inputs')
    sudoku_inputs = sudoku_inputs[sudoku_inputs['value']!=0].copy()

    input_data = []

    for index, row in sudoku_inputs.iterrows():
        input_data.append((row['value'],row['row'],row['column']))

    return input_data


def main():
    input_data = sudoku_inputs()
    problem = sudoku()
    problem.sudoku_solve(input_data)
   
    solution = DataFrame(data = problem.result_vars,
                         index = [i for i in range(1,10)],
                         columns = [i for i in range(1,10)])
    solution.reset_index(inplace=True)
    solution.rename(columns = {'index' : 'filas'},
                    inplace = True)

    solution.to_csv('results.csv',
                     sep=';',
                     index=False)
 
if __name__ == '__main__':
    main() 


