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
    print("Finish search!! The best feature subset is ", bestSet[1], ", which has an accuracy of ", float("{:.2f}".format(bestSet[0]*100)), "%\n")


def main():
    start = time.time()
    
    print("Welcome to CS170 project 2 - Feature Selection using Nearest Neighbor")
    
    validFile = False
    while(not validFile):
        try:
            selection = input("Enter '1' to run a small dataset, '2' to run a large dataset: ")
            if selection == '1':
                fileSize = "Small_Data"
            elif selection == '2':
                fileSize = "Large_Data"

            fileSelection = input("\nEnter the number of dataset you want to run (1-125) : ")
            
            inputFile = 'datasets/CS170_' + fileSize + "__" + fileSelection + ".txt"
            print("Selected dataset: ", inputFile + '\n')
            
            # Using pandas to convert txt to csv
            dataFrame = pd.read_csv(inputFile, header = None, delim_whitespace=True)
            validFile = True
            
        except KeyboardInterrupt:
            print("\n\n\nKeyboardInterrupt")
            print("Exiting the program now...")
            return(1)
        
        except:
            print("Invalid file...")
            print("Please enter again!\n")
            
        
            
    
    print("This dataset has ", dataFrame.shape[1] - 1, " features (not including the class attribute), with ", dataFrame.shape[0], " instances.\n")
    
    algroSelection = input("Enter '1' to run Forward Selection, '2' to run Backward Elimination: ")
    if algroSelection == '1':
        print("\nRunning Forward Selection with all ", dataFrame.shape[1] - 1, " features, using Nearest Neighbor classifier")
        print("Beginning search...")
        forwardSelection(dataFrame)
    else:
        print("\nRunning Backward Elimination with all ", dataFrame.shape[1] - 1, " features, using Nearest Neighbor classifier")
        print("Beginning search...")
        backwardElimination(dataFrame)
    
    end = time.time()
    print("Time taken is ", float("{:.2f}".format(end - start)), " seconds")

if __name__ == "__main__":
    main()
