import sys, os
import sqlalchemy as db
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import arrow
import itertools
value=67
colors=[]
Interval = 26
timeOfDay=480
engine = db.create_engine("mssql+pyodbc://sa:123456@localhost/us2_spts?driver=SQL+Server+Native+Client+11.0")
def allWorker():
    query = '''SELECT[workerID]
    FROM[us2_spts].[wim_spts].[workers]'''
    df = Query_Runner(query)
    return df
def allOrderGetter():
    query="SELECT  [orderID] FROM [us2_spts].[wim_spts].[orders]"
    df=Query_Runner(query)
    return df
def lineOps(orderno):
   query="Select operationAutoID, opsequence ,SMV from [us2_spts].[wim_spts].[stylebulletin] where orderid='"+orderno+"' ORDER BY opsequence"
   df=Query_Runner(query)
   df.set_index('operationAutoID', inplace=True)
   return df
def getAllProgress(orderno):
   query="SELECT * from [us2_spts].dbo.progresscomplete where orderid= '"+orderno +"'"
   df=Query_Runner(query)
   return df
def macIDgetter():
   query="SELECT  [macID],[machineID]   FROM [us2_spts].[wim_spts].[machines]"
   df=Query_Runner(query)
   return df
def lineDataGetter(orderno, lineID, startTime,endTime):
    query1 = '''select  time,[orderID],operationAutoID,lineID,bundleID,workerID,SMV,quantity,SAM
    from [us2_spts].[dbo].[progresscomplete]
    where orderID=''' +"'" +orderno+ "'"+'''and lineID= '''+lineID+''' and time between ''' +"'"+startTime+"'" +''' and  '''+ "'"+endTime+"'"+'''
    order by time'''
    df2 = Query_Runner(query1)
    return df2
def ideal_Matrix_Data_Getter(orderno):
   query1 = '''select t1.orderID, bundleID, operationAutoID, opsequence, SMV * t2.quantity as SAM
   from cfl_spts.orders as t1
   inner join cfl_spts.cutreport as t2 on t1.orderID = t2.orderID
   inner join cfl_spts.stylebulletin as t3 on t2.orderID = t3.orderID
   where t1.orderID = ''' + orderno + '''
   order by t2.bundleID, t3.opsequence
   '''
   df2 = Query_Runner(query1)
   return df2
def hole_Matrix_Data_Getter(orderno):
    query1 = ''' SELECT  [time]
         ,[bundleID]
         ,[operationAutoID]
         ,[macID]
         ,[workerID]
         ,[SMV]
         ,quantity
         ,[SAM]
     FROM us2_spts.dbo.[progresscomplete]
     where orderID=''' + orderno + '''
     order by [time]'''
    df2 = Query_Runner(query1)
    return df2
def OperationAutoIDGetter(orderno):  # Sequence-wise all operation auto id of an order from style bulletin
   query1 = '''  SELECT operationAutoID as 'sortedOps', opsequence, styleID
       FROM [us2_spts].[wim_spts].[stylebulletin]
       where [orderID]=''' + orderno + ''' order by opsequence'''
   df2 = Query_Runner(query1)
   return df2
def allOperationAutoIDGetter():
 query="Select distinct operationautoid from [us2_spts].[wim_spts].stylebulletin"
 df= Query_Runner(query)
 return df["operationautoid"]
def fault_operation_getter():
   query=" SELECT* from [us2_spts].[traffic].[Temp]"
   return Query_Runner(query)
def Query_Runner(query):
   df = pd.read_sql_query(query, engine)
   return df
def timeConverter(currentTime,endTime):   #overtime is taken as SAM
   minDate = arrow.get(currentTime)
   maxDate = arrow.get(endTime)  # datetime.strptime(str(dates[-1]), date_format)
   delta = maxDate - minDate
   minutes=delta.seconds/60
   if(minutes>timeOfDay or delta.days>0 or delta.days<0):
       return -1   #for SAM
   else:
       return minutes
def differenceColumn(df):
   allTime=list(df['time'])
   arr=[]
   for index in range(0,len(allTime)-1):
       y=timeConverter(allTime[index],allTime[index+1])
       if y==-1:
           row=df.iloc[index]
           arr.append(row['SAM']/row['quantity'])
       else:
           row = df.iloc[index]
           arr.append(y/row['quantity'])
   if(len(allTime)!=0):
       row = df.iloc[len(allTime)-1]
       arr.append(row['SAM']/row['quantity'])
   return arr
def MedianTaker(allOperationAutoID ,workerOpList, faultsFrame):
   actual={}
   ideal={}
   efficiency={}
   actualWMVAverage={}
   for index,each in enumerate(allOperationAutoID):
       x=workerOpList[workerOpList['operationAutoID']==each]
       y=faultsFrame[faultsFrame["operationAutoID"]==each]
       if(len(x)!=0):
           actual.update({each:(sum(x['timeDifference']))})
           ideal.update({each:(sum(x['SAM']))})
           actualWMVAverage.update({each:(sum(x['timeDifference'])/len(x['timeDifference']))})
           if len(y)>0:
            efficiency = {key: np.array([ideal[key] - actual.get(key, "#"), y["numOfFaults"].iloc[0]]) for key in ideal.keys()}
           else:
            efficiency = {key: np.array([ideal[key] - actual.get(key, "#"), 0.0]) for key in
                             ideal.keys()}
   return efficiency,ideal,actual,actualWMVAverage
def workerMatrix(orderno, faultsFrame):
   print('Worker Matrix',orderno)
   df2=hole_Matrix_Data_Getter(orderno)
   df=OperationAutoIDGetter(orderno)
   workerEfficiency={}
   idealSam = {}
   actualSam = {}
   actualSamAverage={}
   workers=df2['workerID'].unique()
   for workerID in workers:
       df3=df2[df2['workerID'] == workerID]
       print(df3)
       df3['timeDifference'] = differenceColumn(df3)
       Eff,ideal,actual,SamAverage=MedianTaker(df['sortedOps'].values ,df3,faultsFrame[faultsFrame["workerID"]==workerID])
       workerEfficiency.update({workerID:Eff})
       idealSam.update({workerID:ideal})
       actualSam.update({workerID:actual})
       actualSamAverage.update({workerID:SamAverage})
   return workerEfficiency,idealSam,actualSam,actualSamAverage

def efficiencyMatrix():
    workerFrame=allWorker()
    allOrderFrame=allOrderGetter()
    allOrderList=allOrderFrame["orderID"].values
    allOrderList=['1120180661/8','1120180939/6','1120180940/6','1120180940/7','1120180940/8']
    allWorkerEfficiencies={}
    allIdealSam={}
    allActualSam={}
    actualSamAverage={}
    for key in workerFrame['workerID']:
        allWorkerEfficiencies.update({key:{}})
        allIdealSam.update({key:{}})
        allActualSam.update({key:{}})
        actualSamAverage.update({key:{}})
    alloperations=list(allOperationAutoIDGetter().values)
    alloperations.insert(0, "workerID")
    faultsFrame=fault_operation_getter()
    for order in allOrderList:
        orderno = "'" + str(order) + "'"
        efficiency,ideal,actual,AverageSam=workerMatrix(orderno, faultsFrame[faultsFrame["orderID"]==order])
        for each in efficiency.keys():
            if each in allWorkerEfficiencies.keys():                                                            #means that element is already added so we have to sum up
                allWorkerEfficiencies.update({each:{key: allWorkerEfficiencies[each].get(key, 0) + efficiency[each].get(key, 0)   #adding corresponding worker efficiency
                                         for key in set(allWorkerEfficiencies[each]) | set(efficiency[each])}})
                allIdealSam.update({each:{key: allIdealSam[each].get(key, 0) + ideal[each].get(key, 0)   #adding corresponding worker efficiency
                                         for key in set(allIdealSam[each]) | set(ideal[each])}})
                allActualSam.update({each:{key: allActualSam[each].get(key, 0) + actual[each].get(key, 0)   #adding corresponding worker efficiency
                                         for key in set(allActualSam[each]) | set(actual[each])}})
                actualSamAverage.update({each:{key: actualSamAverage[each].get(key, 0) + AverageSam[each].get(key, 0)   #adding corresponding worker efficiency
                                         for key in set(actualSamAverage[each]) | set(AverageSam[each])}})
            else:
                allWorkerEfficiencies.update({each:efficiency.get(each, 'Not Found')})    #adding worker efficiency in bigger adjacency list
                allIdealSam.update({each:ideal.get(each, 'Not Found')})
                allActualSam.update({each:actual.get(each, 'Not Found')})
                actualSamAverage.update({each: AverageSam.get(each, 'Not Found')})
    nestedDictToCsv(r'C:\Users\Lucky\Desktop\WiMetrix files\Ideal.csv',allIdealSam,alloperations)
    nestedDictToCsv(r'C:\Users\Lucky\Desktop\WiMetrix files\Actual.csv', allActualSam, alloperations)
    nestedDictToCsv(r'C:\Users\Lucky\Desktop\WiMetrix files\WorkerEfficiency.csv', allWorkerEfficiencies, alloperations)
    nestedDictToCsv(r'C:\Users\Lucky\Desktop\WiMetrix files\SamAverage.csv', actualSamAverage, alloperations)
def nestedDictToCsv(fileName,Dict,alloperations):
    with open(fileName, "w", newline ='') as f:
        w = csv.DictWriter(f, alloperations)
        w.writeheader()
        for key, val in (Dict.items()):
            row = {'workerID': key}
            row.update(val)
            w.writerow(row)
def operationConverter(refTable,x):
   arr = []
   opsequence = list(refTable["opsequence"])
   autoid = list(refTable["operationAutoID"])
   i = 0
   while i != len(autoid):
       j = i + 1
       arr1 = []
       arr1.append(autoid[i])
       while j != len(autoid) and opsequence[i] == opsequence[j]:
           arr1.append(autoid[j])
           j = j + 1
       i = j
       arr.append(arr1)
   i = 0
   j = 0
   arr3 = []
   arr4=[]
   while (j != len(x) and i != len(arr)):
       try:
           index = arr[i].index(x[j])
           arr3.append(x[j])
           j = j + 1

       except:
           i = i + 1
   return arr3,arr4
def lineBalancing(orderno):
   progressFrame=getAllProgress(orderno)
   progressFrame["opsequence"]=progressFrame["opsequence"].fillna(1)
   machineFrame=macIDgetter()
   opFrame=lineOps(orderno)
   mainFrame = pd.DataFrame(index=machineFrame["macID"].values, columns=opFrame["opsequence"].unique())
   opFrame["opsequence"] = opFrame["opsequence"].fillna(1)
   for row in machineFrame["macID"]:
      df=progressFrame[progressFrame['macID'] == row]
      #print(row, list(df['operationAutoID'].unique()))
      if len(df)!=0:
           x,sequence=operationConverter(opFrame, list(df['operationAutoID'].unique()))
           print(x,sequence)
           mainFrame.set_value(row,sequence[0],x)
           print(row,x)
def IdealBalance(orderno):
    opFrame= lineOps(orderno)
    #query="SELECT top 1 max([quantity]) as quantity from [us2_spts].[wim_spts].[cutreport] where orderid='" +orderno+"'group by bundleid order by max([quantity]) desc"
    #df=Query_Runner(query)
    #max=df["quantity"].values
    arr=[]
    for i in opFrame["SMV"].values :
        arr.append(i)#*max[0])
    CL= np.max(np.array(arr))            # Worst Cycle Time of assembly Line
    Tp= sum(arr)   #Total Processing Time
    n= len(opFrame)
    DL= ((n*CL - Tp)/(n*CL ))*100 # balance delay
    PL= 9*60/CL # production
    print('Production ',PL, Tp, n)
    print(DL)
    avgTimeTaken= Tp/n
    print(avgTimeTaken)
    return CL ,opFrame
def OptimizedBalance(orderno):
   actualSamAverage, allWorkerEfficiencies = efficiencyMatrix()
   maxVal, opFrame= IdealBalance(orderno)
   df2 = hole_Matrix_Data_Getter("'"+orderno+"'")
   LBRefTable=pd.DataFrame(index=df2["workerID"].unique(),columns=opFrame['operationAutoID'].values)
   for  op in  list(LBRefTable.columns):
       for index in list(df2["workerID"].unique()):
               try:
                   if(actualSamAverage[index][op]):
                        LBRefTable.set_value(index,op, maxVal- actualSamAverage[index][op])
               except:
                    continue
   print(LBRefTable)
   #LBRefTable.to_csv(r'C:\Users\LUCKY\Desktop\9) Nested Dictionaries for Worker Efficiencies\xyz.csv')
def machineCalculator(target, efficiency, availableTime, opFrame):
    arr=np.divide(np.divide(np.array([target*val for val in opFrame['SMV'].values]),availableTime),efficiency)
    machines=[]
    for index,val in enumerate(arr):
        if(val-int(val)>=0.2):
            machines.append(math.ceil(val))
        else:
            machines.append(math.floor(val))
    return machines

def findsubsets(setOfValues, size):    #return List of subsets with specific size n
    return list(map(list, itertools.combinations(setOfValues, size)))

def valueTaker(allOperationAutoID, df2):
    actual = {}
    ideal = {}
    efficiency = {}
    actualSamAverage = {}
    for index, each in enumerate(allOperationAutoID):
        x = df2[df2['operationAutoID'] == each]
        if (len(x) != 0):
            actual.update({each: (sum(x['timeDifference']))})
            ideal.update({each: (sum(x['SAM']))})
            actualSamAverage.update({each: (sum(x['timeDifference']) / len(x['timeDifference']))})
            efficiency = {
            key: np.array([ideal[key] - actual.get(key, "#"), (actual.get(key, 0) / ideal[key]) * 100]) for key in
            ideal.keys()}
    return efficiency, ideal, actual, actualSamAverage


def TimeAnalysis(orderno, lineID, startTime, endTime):
    df2 = lineDataGetter(orderno, lineID, startTime, endTime)
    df = OperationAutoIDGetter("'" + orderno + "'")
    workerEfficiency = {}
    idealSam = {}
    actualSam = {}
    actualSamAverage = {}
    workers = df2['workerID'].unique()
    for workerID in workers:
        df3 = df2[df2['workerID'] == workerID]
        df3['timeDifference'] = differenceColumn(df3)
        Eff, ideal, actual, SamAverage = valueTaker(df['sortedOps'].values, df3)
        workerEfficiency.update({workerID: Eff})
        idealSam.update({workerID: ideal})
        actualSam.update({workerID: actual})
        actualSamAverage.update({workerID: SamAverage})
    print('Worker Efficiecy ', workerEfficiency)
    print('Ideal Operational Efficiency ', idealSam)
    print('Actual SAM ', actualSam)
    print('Actual SAM per Bundles', actualSamAverage)
    return workerEfficiency, idealSam, actualSam, actualSamAverage


def LineComputation(target,opFrame,HoleFrame,approachingValue,threshold,flag):
    opFrame['opsequence'].fillna(1, inplace=True) # Fill null entries in opsequence column as 1
    Line = pd.DataFrame(index= opFrame.index)# it will contain worker Ids... for mapping a worker to operation
    occupiedWorkers=[]
    for operation,row in opFrame.iterrows():
        op=str(operation)
        machinesOfOperation=row['Machine Quantity']
        if(flag==0):
            thresholdFrame = HoleFrame[HoleFrame[op].between(approachingValue - (approachingValue / threshold), approachingValue)]
        else:
            thresholdFrame = HoleFrame[HoleFrame[op].between(approachingValue, approachingValue +(approachingValue / threshold))]
        if not thresholdFrame.empty:
            thresholdFrame[op]=np.abs(approachingValue - np.array(thresholdFrame[op].values))
            sortedFrame = thresholdFrame.sort_values(by=op)
            arr=[]
            tempApproach=0
            takenMachines=0
            for workerID, value in sortedFrame.iterrows():
                if workerID not in occupiedWorkers:
                    if tempApproach<=target and takenMachines<=machinesOfOperation:
                        wmv = HoleFrame.loc[workerID,op]  # data from origional sam average frame
                        numberOfPieces=timeOfDay/math.ceil(wmv)
                        tempApproach=tempApproach+numberOfPieces
                        takenMachines = takenMachines+1
                        arr.append(workerID)
                        occupiedWorkers.append(workerID)
                    else:
                        break

                else:
                    print('best Same Worker Found again',workerID)

            Line.set_value(operation, 'WorkerID', arr)
        else:
            print('No worker Found in Threshold')
    return Line
def OptimizedTechnique(orderno):
   approachingValue, opFrame= IdealBalance(orderno)
   print("Approaching Value: ",approachingValue)
   #print('Enter Line Target : ')
   target=2500
   target=np.float(target)
   efficiency=0.8
   availableTime=timeOfDay
   opFrame['Machine Quantity']=machineCalculator(target, efficiency, availableTime, opFrame)
   actualSamFrame=pd.read_csv(r'C:\Users\Lucky\Desktop\WiMetrix files\SamAverage.csv', index_col=0)
   counter=4
   flag =0
   for i in range(1,9):
       if(i>=1 and i<=4):
            if counter==1:
                counter=4/3
            Line=LineComputation(target,opFrame,actualSamFrame,approachingValue,counter,0)
       else:
           if flag==0:
               counter=4
               flag=1
           if counter == 1:
                counter=4/3
           Line=LineComputation(target,opFrame,actualSamFrame,approachingValue,counter,1)
       print(Line)
       counter=counter-1

       print(Line)
   #LBRefTable.to_csv(r'C:\Users\LUCKY\Desktop\9) Nested Dictionaries for Worker Efficiencies\xyz.csv')


def operationWiseSubsetMaker(orderno):
    approachingValue, opFrame = IdealBalance(orderno)
    target = 2500
    target = np.float(target)
    efficiency = 0.8
    availableTime = timeOfDay
    opFrame['Machine Quantity'] = machineCalculator(target, efficiency, availableTime, opFrame)
    actualSamFrame = pd.read_csv(r'C:\Users\Lucky\Desktop\WiMetrix files\SamAverage.csv', index_col=0)
    operationAutoID=opFrame['operationAutoID'].values
    print(actualSamFrame)
    print(opFrame)
    operationWiseSubsets={}
    # fileopen = open('data.txt', 'w+')
    for each in operationAutoID:
        arrOfSets = []
        machinesOfOperation=opFrame[opFrame['operationAutoID']==each]['Machine Quantity'].values[0]
        print('operation', each,'Machines ',machinesOfOperation)
        # fileopen.write('operation: {}, Machines: {}\n\n'.format(each, machinesOfOperation))
        each = str(each)
        workersOfOperation=actualSamFrame[actualSamFrame[each].notnull()][each]
        for i in range(1,machinesOfOperation+1):
            arrOfSets=arrOfSets+findsubsets(set(workersOfOperation.index.values),i)
        operationWiseSubsets.update({each:arrOfSets})
        print("No of Workers: ",len(workersOfOperation),"Subset size ",len(operationWiseSubsets[each])," ",operationWiseSubsets[each])
        # fileopen.write('No of Workers: {}, Subset size: {} {}\n------------------------------\n\n'.
        #                format(len(workersOfOperation), len(operationWiseSubsets[each]),
        #                       operationWiseSubsets[each]))

    # fileopen.close()
    return operationWiseSubsets

def subsetComputation(dictOfSubsets):
    AverageSamFrame = pd.read_csv(r'C:\Users\Lucky\Desktop\WiMetrix files\SamAverage.csv', index_col=0)
    setWiseValues=[]
    for operation in dictOfSubsets.keys():
        sets=dictOfSubsets[operation]
        for eachSet in sets:  #OneD array of workers
            filteredFrame=AverageSamFrame[AverageSamFrame.index.isin(eachSet)]
            setWiseValues.append(max(filteredFrame[operation].values))
        sort=[x for _, x in sorted(zip(setWiseValues, sets))]
        dictOfSubsets.update({operation:sort})

    print(dictOfSubsets)




try:
  #lineBalancing('1120180661/8')
  #IdealBalance('1120180661/8')
  #OptimizedBalance('1120180661/8')
  efficiencyMatrix()
  #OptimizedTechnique('1120180661/8')
  #dictOfSubsets=operationWiseSubsetMaker('1120180661/8')
  #subsetComputation(dictOfSubsets)
  #TimeAnalysis('2120181982/1','1','2018-12-29 14:00:00','2018-12-29 18:00:00')


except Exception as e:
  print('Exception Occured ' + str(e))
  exc_type, exc_obj, exc_tb = sys.exc_info()
  fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
  print(exc_type, fname, exc_tb.tb_lineno)



# select t2.*, t3.*
#  INTO [us2_spts].[traffic].[Temp]
#  from traffic.qualitylog as t1
#  inner join (
#   SELECT pC.workerID, max(qualityLogID) as qualityLogID,
#       [qualityFault_operationID]
#       ,sum([defectsNo]) as numOfFaults
#   FROM [us2_spts].[traffic].[qualitylog] as qL
#   LEFT JOIN us2_spts.dbo.[progresscomplete] as pC
#   ON qL.qualityFault_operationID = pC.styleID and qL.itemID=pC.itemID
#   where qL.qualityFaultID < 1000 and qL.rework = 0
#   group by workerID, [qualityFault_operationID]
# ) as t2 on t1.qualityLogID = t2.qualityLogID
# inner join wim_spts.stylebulletin as t3 on t1.qualityFault_operationID = t3.styleID
# where t2.workerID IS NOT NULL
#  --having qL.qualityFaultID < 1000
















