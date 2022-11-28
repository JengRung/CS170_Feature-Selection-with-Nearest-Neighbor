import pandas as pd

# Using pandas to convert txt to csv
inputFile = 'datasets/CS170_Small_Data__96.txt' 
dataFrame = pd.read_csv(inputFile, header = None, delim_whitespace=True,)

def nearestNeighbor(df, indexList):
    '''
    nearest neighbor process:
    Iterate through the list of each attribute
    Calculate the distance between the current point and all other points
    Store the distance and the index of the point in a list
    Find the minimum distance and the index of the point
    '''
    size = len(df)
    correctCount = 0
    for i in range(size):
        # Store the distance and the index of the point in a list
        distance = []
        for j in range(size):
            if i != j:
                # Calculate the distance between the current point and all other points
                dist = 0
                for k in indexList:
                    dist += (df[k][i] - df[k][j]) ** 2
                dist = dist ** 0.5
                distance.append([dist, j])
        # Find the minimum distance and the index of the point
        minDist = min(distance)
        # Check if the type of the point is the same as the type of the nearest neighbor
        if df[0][i] == df[0][minDist[1]]:
            correctCount += 1
        # # Print the index of the point and the distance
        # print(i, minDist[1], minDist[0])

    print("Accuracy: ", correctCount/size)

nearestNeighbor(dataFrame, [1, 3, 6])