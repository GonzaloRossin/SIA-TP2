# x: entrance data
# y: expected output


def data(input):
    x =[]
    y=[]
    
    if input.upper() == "AND":
        x = [[-1, 1],
             [1, -1],
             [-1, -1],
             [1, 1]
        ]
        y = [-1, -1, -1, 1]
    elif input.upper() == "XOR":
        x = [[-1, 1],
             [1, -1],
             [-1, -1],
             [1, 1]
        ]
        y = [1, 1, -1, -1]
        
    
    return x, y
