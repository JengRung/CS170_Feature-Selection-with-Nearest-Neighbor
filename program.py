import pandas as pd
import time

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

    accuracy = correctCount / size
    print("Using feature(s) ", indexList, " accuracy is ", float("{:.2f}".format(accuracy*100)), "%")
    return accuracy

def forwardSelection(df):
    # Current array append the best feature (1) from each iteration, increase until the size match the number of attributes
    curArr = []
    # Best array store the best feature set (1+) from each iteration, return the best feature set at the end
    bestArr = []
    # Number of attributes
    rowSize = df.shape[1]
    while(len(curArr) < rowSize - 1):
        result = []
        for i in range(1, rowSize, 1):
            # Call the nearestNeighbor function
            if i not in curArr:
                tempArr = []
                tempArr.extend(curArr)
                tempArr.append(i)
                result.append([nearestNeighbor(df, tempArr), i])
        best = max(result)
        
        bestSet = []
        bestSet.extend(curArr)
        bestSet.extend([best[1]])
        bestArr.append([best[0] ,bestSet])
        
        curArr.append(best[1])
        print("Feature set ", bestSet, " was best, accuracy is ", float("{:.2f}".format(best[0]*100)), "%\n")

    bestSet = max(bestArr)
    print("Finish search!! The best feature subset is ", bestSet[1], ", which has an accuracy of ", float("{:.2f}".format(bestSet[0]*100)), "%")
    

def backwardElimination(df):
    # Current array append the best feature (1) from each iteration, decrease until the size match 0
    curArr = []
    # Best array store the best feature set (1+) from each iteration, return the best feature set at the end
    bestArr = []
    # Number of attributes
    rowSize = df.shape[1]
    for i in range(1, rowSize, 1):
        curArr.append(i)
    
    # Test all the subsets
    allSetAccuracy = nearestNeighbor(df, curArr)
    bestArr.append([allSetAccuracy, curArr])
    print("Feature set ", curArr, " was best, accuracy is ", float("{:.2f}".format(allSetAccuracy*100)), "%\n")
    
    # Start to eliminate the feature until the size reach 0
    while(len(curArr) != 0):
        result = []
        for i in range(1, rowSize, 1):
            # Call the nearestNeighbor function
            if i in curArr:
                tempArr = []
                tempArr.extend(curArr)
                tempArr.remove(i)
                result.append([nearestNeighbor(df, tempArr), i])
        best = max(result)
        
        bestSet = []
        bestSet.extend(curArr)
        bestSet.remove(best[1])
        bestArr.append([best[0] ,bestSet])
        
        curArr.remove(best[1])
        print("Feature set ", bestSet, " was best, accuracy is ", float("{:.2f}".format(best[0]*100)), "%\n")

    bestSet = max(bestArr)
    print("Finish search!! The best feature subset is ", bestSet[1], ", which has an accuracy of ", float("{:.2f}".format(bestSet[0]*100)), "%")


def main():
    start = time.time()
    
    # inputFile = 'datasets/CS170_Small_Data__6.txt' 
    inputFile = 'datasets/CS170_Large_Data__96.txt' 
    # Using pandas to convert txt to csv
    dataFrame = pd.read_csv(inputFile, header = None, delim_whitespace=True,)
    
    forwardSelection(dataFrame)
    # backwardElimination(dataFrame)
    
    end = time.time()
    print("Time taken is ", float("{:.2f}".format(end - start)), " seconds")

if __name__ == "__main__":
    main()
