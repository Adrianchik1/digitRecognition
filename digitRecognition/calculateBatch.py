

def calculateBatch(X, y, batchSize, iteration):
    start_index = (iteration * batchSize) % len(X)
    end_index = start_index + batchSize
    
    if end_index <= len(X):
        batchX = X[start_index:end_index]
        batchY = y[start_index:end_index]
    else:
        batchX = X[start_index:] + X[:end_index % len(X)]
        batchY = y[start_index:] + y[:end_index % len(y)]
    return batchX, batchY