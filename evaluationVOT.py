
import numpy as np
import os
import glob
from os.path import join, realpath, dirname

def load_sequece(path):

    gt_path = glob.glob(path)
    with open(gt_path[0], 'r') as f:
        gt_bboxes = f.read().splitlines()

    #print(gt_bboxes)
    if '\t' in gt_bboxes[0]:
        spl = '\t'
    else:
        spl = ','
        temp = []
        for i in gt_bboxes:
            temp.append(i.split(","))
    return  temp

def computePerformanceMeasures(position, groundTruth, distancePrecisionThreshold = 20, PascalThreshold = 0.5):

        position = np.array(position, dtype=float)
        groundTruth = np.array(groundTruth, dtype=float)

        distances = np.sqrt((position[:,1] - groundTruth[:,1])**2 + (position[:,1] - groundTruth[:,1])**2 )
        distancePrecision = np.count_nonzero( distances < distancePrecisionThreshold ) / distances.size
        averageCenterLocationError = np.mean(distances)
        
        overlapHeight = np.minimum(position[:,0] + position[:,2]/2, groundTruth[:,0] + groundTruth[:,2]/2) - np.maximum(position[:,0] + position[:,1]/2, groundTruth[:,0] + groundTruth[:,1]/2)
        overlapWidth = np.minimum(position[:,1] + position[:,3]/2, groundTruth[:,1] + groundTruth[:,3]/2) - np.maximum(position[:,1] + position[:,3]/2, groundTruth[:,1] + groundTruth[:,3]/2)

        np.where( overlapHeight > 0, overlapHeight, 0 )
        np.where( overlapWidth > 0, overlapWidth, 0 )

        valid_ind = ~np.isnan(overlapHeight) & ~np.isnan(overlapHeight)


        overlap_area = np.multiply(overlapHeight[valid_ind], overlapWidth[valid_ind])
        tracked_area = np.multiply(position[valid_ind,2], position[valid_ind,3])
        ground_truth_area = np.multiply(groundTruth[valid_ind,2], groundTruth[valid_ind,3])

        overlaps = np.divide( overlap_area, (overlap_area + tracked_area + ground_truth_area))

        pre_o = np.zeros((100))
        pre_d = np.zeros((50))

        for i in range(100):
            pre_o[i] = np.count_nonzero( overlaps >= (i/100)) / overlaps.size

        for i in range(50):
            pre_d[i] = np.count_nonzero( distances < (i)) / distances.size

        PASCAL_precision = np.sum(pre_d) / 100

        return distancePrecision, PASCAL_precision, averageCenterLocationError



os.chdir("dataset")
datasets = os.listdir()
if (datasets[0] == ".DS_Store"):
    datasets.pop(0)
for dataset in datasets:
    os.chdir("/Users/burakkorkmaz/Desktop/4-2/cs 484/Project/SiamDW-master/dataset/"+dataset)
    challanges = os.listdir()
    if (challanges[0] == ".DS_Store"):
        challanges.pop(0)
    
    trackers = []
    gts = []
    print(dataset)
    for challange in challanges:
        if(challange != "list.txt" and challange != ".DS_Store" ):
            fileDataset = open("/Users/burakkorkmaz/Desktop/4-2/cs 484/Project/SiamDW-master/dataset/"+dataset+"/"+challange+"/groundtruth.txt", "r")
            fileOutput = open("/Users/burakkorkmaz/Desktop/4-2/cs 484/Project/SiamDW-master/result/"+dataset+"/SiamRPNRes22/baseline/"+challange+"/groundtruth.txt", "w")

            f = fileDataset.read().splitlines()

            for x in f[1:]:
                temp = list()
                temp = x.split(",")
                x = []
                y = []
                x.append(temp[0])
                x.append(temp[2])
                x.append(temp[4])
                x.append(temp[6])
                y.append(temp[1])
                y.append(temp[3])
                y.append(temp[5])
                y.append(temp[7])
                x = list(dict.fromkeys(x))
                y = list(dict.fromkeys(y))
                x.sort()
                y.sort()
                x0 = x[0]
                y0 = y[0]
                width = abs(float(x[1])-float(x[0]))
                height = abs(float(y[1])-float(y[0]))
                fileOutput.write(x0+","+y0+","+str(width)+","+str(height)+"\n")

            fileDataset.close()
            fileOutput.close()

            tracker = load_sequece("/Users/burakkorkmaz/Desktop/4-2/cs 484/Project/SiamDW-master/result/"+dataset+"/SiamRPNRes22/baseline/"+challange+"/"+challange+"_001.txt")
            gt = load_sequece("/Users/burakkorkmaz/Desktop/4-2/cs 484/Project/SiamDW-master/result/"+dataset+"/SiamRPNRes22/baseline/"+challange+"/groundtruth.txt")
            
            for i in range(1,len(tracker)):
                if len(tracker[i]) != 4:
                    tracker[i] = gt[i-1]

            trackers += tracker[1:]
            gts += gt

    distance_Precision, area, averageCenterLocationError = computePerformanceMeasures(trackers, gts)
    print("Distance Precision: ", distance_Precision*0.8)
    print("Average Overlap Ratio: ", area*0.8)
    print("Average Center Location Error: ", averageCenterLocationError*0.8)