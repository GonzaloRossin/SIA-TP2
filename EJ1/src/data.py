# x: entrance data
# y: expected output


def data(input):
    input.upper()
    
    if input == "AND":
        x = [[-1, 1]
             [1, -1]
             [-1, -1]
             [1, 1]
        ]
        y = [-1, -1, -1, 1]
    elif input == "XOR":
        x = [[-1, 1]
             [1, -1]
             [-1, -1]
             [1, 1]
        ]
        y = [1, 1, -1, -1]
        
    
    return x, y

#sys.argv[0]