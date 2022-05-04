#Importing Libraries
from re import A
from turtle import end_fill
import h5py
import numpy as np                
import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib import animation
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display


#Initializing File Read
filename = "scalarHIT_fields100.h5"

f = h5py.File("scalarHIT_fields100.h5",'r')
list(f.keys())
dset = f['fields']



def run_case(case,casename,skip):
    if skip == 'mean_':
        #taking mean of every 2 points
        outputarray_2 = np.zeros(shape=(100,64,64,64))
        for n in range(0,100):
            print('Calculating 2 mean filter:' + str(n) + "%",end = "\r")
            for i in range(0,128,2):
                for j in range(0,128,2):
                    for k in range(0,128,2):
                        outputarray_2[n,i//2,j//2,k//2] = (dset[n,i,j,k,case] + dset[n,i+1,j,k,case] + dset[n,i,j+1,k,case] + dset[n,i,j,k+1,case])/4.0
        print("done with 2 filter")

#         #taking mean of every 4 points
#         outputarray_4 = np.zeros(shape=(100,32,32,32))
#         for n in range(0,100):
#             print('Calculating 4 mean filter:' + str(n) + "%",end = "\r")
#             for i in range(0,128,4):
#                 for j in range(0,128,4):
#                     for k in range(0,128,4):
#                         outputarray_4[n,i//4,j//4,k//4] = (dset[n,i,j,k,case] + dset[n,i+1,j,k,case] + dset[n,i,j+1,k,case] + dset[n,i,j,k+1,case] \
#                         + dset[n,i+2,j,k,case] + dset[n,i,j+2,k,case] + dset[n,i,j,k+2,case] + dset[n,i+3,j,k,case] + dset[n,i,j+3,k,case] + dset[n,i,j,k+3,case]) / 10
#         print("done with 4 filter")

#         #taking mean of every 8 points
#         outputarray_8 = np.zeros(shape=(100,16,16,16))
#         for n in range(0,100):
#             print('Calculating 8 mean filter:' + str(n) + "%",end = "\r")
#             for i in range(0,128,8):
#                 for j in range(0,128,8):
#                     for k in range(0,128,8):
#                         outputarray_8[n,i//8,j//8,k//8] = (dset[n,i,j,k,case] + dset[n,i+1,j,k,case] + dset[n,i,j+1,k,case] + dset[n,i,j,k+1,case] \
#                         + dset[n,i+2,j,k,case] + dset[n,i,j+2,k,case] + dset[n,i,j,k+2,case] + dset[n,i+3,j,k,case] + dset[n,i,j+3,k,case] + dset[n,i,j,k+3,case]) \
#                         + dset[n,i+4,j,k,case] + dset[n,i,j+4,k,case] + dset[n,i,j,k+4,case] + dset[n,i+5,j,k,case] + dset[n,i,j+5,k,case] + dset[n,i,j,k+5,case]\
#                         + dset[n,i+6,j,k,case] + dset[n,i,j+6,k,case] + dset[n,i,j,k+6,case] + dset[n,i+7,j,k,case] + dset[n,i,j+7,k,case] + dset[n,i,j,k+7,case]/ 22
#         print("done with 8 filter")
    elif skip == 'skip_':

        skip_filter2 = np.zeros(shape =(100,64,64,64))
        for n in range(0,100):
            print('Calculating 2 skip filter:' + str(n) + "%",end = "\r")
            for i in range(0,128,2):
                for j in range(0,128,2):
                    for k in range(0,128,2):
                        skip_filter2[n,i//2,j//2,k//2] = dset[n,i,j,k,case]
        print('\n')

#         skip_filter4 = np.zeros(shape =(100,32,32,32))
#         for n in range(0,100):
#             print('Calculating 4 skip filter:' + str(n) + "%",end = "\r")
#             for i in range(0,128,4):
#                 for j in range(0,128,4):
#                     for k in range(0,128,4):
#                         skip_filter4[n,i//4,j//4,k//4] = dset[n,i,j,k,case]
#         print('\n')

#         skip_filter8 = np.zeros(shape =(100,16,16,16))
#         for n in range(0,100):
#             print('Calculating 8 skip filter:' + str(n) + "%",end = "\r")
#             for i in range(0,128,8):
#                 for j in range(0,128,8):
#                     for k in range(0,128,8):
#                         skip_filter8[n,i//8,j//8,k//8] = dset[n,i,j,k,case]
#         print('\n')
    else:
        print("incorrect case for skip")
    #Analyzing data

#     #Preallocating storage
#     meanval   = np.zeros(100)
#     varval    = np.zeros(100)
#     skewval   = np.zeros(100)
#     kurtval   = np.zeros(100)
#     meanval_2 = np.zeros(100)
#     varval_2  = np.zeros(100)
#     skewval_2 = np.zeros(100)
#     kurtval_2 = np.zeros(100)
#     meanval_4 = np.zeros(100)
#     varval_4  = np.zeros(100)
#     skewval_4 = np.zeros(100)
#     kurtval_4 = np.zeros(100)
#     meanval_8 = np.zeros(100)
#     varval_8  = np.zeros(100)
#     skewval_8 = np.zeros(100)
#     kurtval_8 = np.zeros(100)
#     iter = 0

    if skip == 'skip_':
        outputarray_2 = skip_filter2
#         outputarray_4 = skip_filter4
#         outputarray_8 = skip_filter8
#     #Calculating mean, var, skew, and kurt for unfiltered
#     for j in range(0,100):
#             iter = iter+1
#             meanval[j]   = dset[j,:,:,:,case].mean()                                           #mean
#             varval[j]    = dset[j,:,:,:,case].var()                                            #variance
#             skewval[j]   = np.mean(np.power(dset[j,:,:,:,case],3)/(np.power(varval[j],(3/2)))) #skewness
#             kurtval[j]   = np.mean(np.power(dset[j,:,:,:,case],4)/(np.power(varval[j],(2))))   #kurtosis
#             print('Analyzing unfiltered data:' + str(iter) + "%", end = "\r")

#     #Calculating mean, var, skew, and kurt for two times filter
#     iter = 0
#     for j in range(0,100):
#             iter = iter+1
#             meanval_2[j]   = outputarray_2[j,:,:,:].mean()                                           #mean
#             varval_2[j]    = outputarray_2[j,:,:,:].var()                                            #variance
#             skewval_2[j]   = np.mean(np.power(outputarray_2[j,:,:,:],3)/(np.power(varval_2[j],(3/2)))) #skewness
#             kurtval_2[j]   = np.mean(np.power(outputarray_2[j,:,:,:],4)/(np.power(varval_2[j],(2))))   #kurtosis
#             print('Analyzing 2 times filter data:' + str(iter) + "%")

#     iter = 0

#     #Calculating mean, var, skew, and kurt for four times filter
#     for j in range(0,100):
#             iter = iter+1
#             meanval_4[j]   = outputarray_4[j,:,:,:].mean()                                           #mean
#             varval_4[j]    = outputarray_4[j,:,:,:].var()                                            #variance
#             skewval_4[j]   = np.mean(np.power(outputarray_4[j,:,:,:],3)/(np.power(varval_4[j],(3/2)))) #skewness
#             kurtval_4[j]   = np.mean(np.power(outputarray_4[j,:,:,:],4)/(np.power(varval_4[j],(2))))   #kurtosis
#             print('Analyzing 4 times filter data:' + str(iter) + "%")
#     iter = 0
#     #Calculating mean, var, skew, and kurt for eight times filter
#     for j in range(0,100):
#             iter = iter+1
#             meanval_8[j]   = outputarray_8[j,:,:,:].mean()                                           #mean
#             varval_8[j]    = outputarray_8[j,:,:,:].var()                                            #variance
#             skewval_8[j]   = np.mean(np.power(outputarray_8[j,:,:,:],3)/(np.power(varval_8[j],(3/2)))) #skewness
#             kurtval_8[j]   = np.mean(np.power(outputarray_8[j,:,:,:],4)/(np.power(varval_8[j],(2))))   #kurtosis
#             print('Analyzing 8 times filter data:' + str(iter) + "%")
#     #converting matrices to one row
#     meanval = meanval.ravel()
#     varval  = varval.ravel()
#     skewval = skewval.ravel()
#     kurtval = kurtval.ravel()

#     meanval_2 = meanval_2.ravel()
#     varval_2  = varval_2.ravel()
#     skewval_2 = skewval_2.ravel()
#     kurtval_2 = kurtval_2.ravel()

#     meanval_4 = meanval_4.ravel()
#     varval_4  = varval_4.ravel()
#     skewval_4 = skewval_4.ravel()
#     kurtval_4 = kurtval_4.ravel()

#     meanval_8 = meanval_8.ravel()
#     varval_8  = varval_8.ravel()
#     skewval_8 = skewval_8.ravel()
#     kurtval_8 = kurtval_8.ravel()

#     #Converting data to Pandas dataframe and transposing
#     data_columns = ['unfiltered: ' , 'two filter: ' , 'four filter: ' , 'eight filter: ']
#     unfiltered_data_mean = [meanval,meanval_2,meanval_4,meanval_8]
#     unfiltered_df_mean = pd.DataFrame(unfiltered_data_mean,index=data_columns).T

#     unfiltered_data_kurt = [kurtval,kurtval_2,kurtval_4,kurtval_8]
#     unfiltered_df_kurt = pd.DataFrame(unfiltered_data_kurt,index=data_columns).T

#     unfiltered_data_skew = [skewval,skewval_2,skewval_4,skewval_8]
#     unfiltered_df_skew = pd.DataFrame(unfiltered_data_skew,index=data_columns).T

#     unfiltered_data_var = [varval,varval_2,varval_4,varval_8]
#     unfiltered_df_var = pd.DataFrame(unfiltered_data_var,index=data_columns).T

#     #writing vals to new file
#     with open(skip + 'turbvals' + casename + '.txt', 'w') as f:
#         f.write("Meanval:\n\n")
#         unfilteredToString = unfiltered_df_mean.to_string(header= True, index= True)
#         f.write(unfilteredToString)
#         f.write("\n\nKurtval:\n\n")
#         unfilteredToString2 = unfiltered_df_kurt.to_string(header= True, index= True)
#         f.write(unfilteredToString2)
#         f.write("\n\nSkewval:\n\n")
#         unfilteredToString4 = unfiltered_df_skew.to_string(header= True, index= True)
#         f.write(unfilteredToString4)
#         f.write("\n\nVarval:\n\n")
#         unfilteredToString4 = unfiltered_df_var.to_string(header= True, index= True)
#         f.write(unfilteredToString4)
#     #Animation subplots
#     #fig, (ax1,ax2,ax3,ax4) = plt.subplots(1,4)
#     #Initializing heatmap anim for colorbar
#     #sb.heatmap(dset[0,:,:,0,case],ax =ax1,square = True,cbar_kws={"shrink": 0.30})
#     #sb.heatmap(outputarray_2[0,:,:,0],ax = ax2,square = True,cbar_kws={"shrink": 0.30})
#     #sb.heatmap(outputarray_4[0,:,:,0], ax = ax3,square = True,cbar_kws={"shrink": 0.30})
#     #sb.heatmap(outputarray_8[0,:,:,0], ax = ax4,square = True,cbar_kws={"shrink": 0.30})
#     #plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace= 0.5, hspace=None)
#     #animating heatmap
#     #def animate(i):
#     #    print(i)
#     #    sb.heatmap(dset[i,:,:,1,case], cbar = False,ax = ax1,square = True)
#     #    sb.heatmap(outputarray_2[i,:,:,1], cbar = False, ax = ax2,square = True)
#     #    sb.heatmap(outputarray_4[i,:,:,1], cbar = False, ax = ax3,square = True)
#     #    sb.heatmap(outputarray_8[i,:,:,1], cbar = False, ax = ax4,square = True)
#     #    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace= 0.5, hspace=None)
    
#     #anim = animation.FuncAnimation(fig,animate,frames = 100, repeat = False)
#     #plt.show(block = False)

#     #saving heatmap anim
#     #writergif = animation.PillowWriter(fps=8) 
#     #anim.save('turb_flow' + casename + '.gif', writer=writergif)
#     #print("done saving anim")

    #Animation Plots
    fig = plt.figure()
    sb.heatmap(dset[0,:,:,0,case],square = True)
    plt.title('unfiltered' + ' Heatmap')
    def animate(i):
        print('Animating unfiltered:' + str(i) + "%",end = "\r")
        sb.heatmap(dset[i,:,:,0,case], cbar = False,square = True)

    anim = animation.FuncAnimation(fig,animate,frames = 100, repeat = False)
    plt.show(block = False)

    writergif = animation.PillowWriter(fps=8) 
    anim.save('unfiltered' + casename + '.gif', writer=writergif)
    print('\n')
    #swithcing modes of filtration
    if skip == 'mean_':

        class AnimMeanHeatmap:
            def __init__(self,filterType,data):
                self.filterType = filterType
                self.data = data

            def animFunc(self):
                fig = plt.figure()
                sb.heatmap(self.data[0,:,:,0],square = True)
                plt.title(skip + self.filterType + ' Heatmap')
                def animate(i):
                    print('Animating ' + self.filterType + ':' + str(i) + "%",end = "\r")
                    sb.heatmap(self.data[i,:,:,0], cbar = False,square = True)

                anim = animation.FuncAnimation(fig,animate,frames = 100, repeat = False)
                plt.show(block = False)

                writergif = animation.PillowWriter(fps=8) 
                anim.save(self.filterType + skip + casename + '.gif', writer=writergif)
                print('\n')
                print("done saving anim")
        twoMeanFilter = AnimMeanHeatmap("2_filter",outputarray_2)
        twoMeanFilter.animFunc()
#         fourMeanFilter = AnimMeanHeatmap("4_filter",outputarray_4)
#         fourMeanFilter.animFunc()
#         eightMeanFilter = AnimMeanHeatmap("eight_filter",outputarray_8)
#         eightMeanFilter.animFunc()

    elif skip == 'skip_':

        class AnimMeanHeatmap:
            def __init__(self,filterType,data):
                self.filterType = filterType
                self.data = data

            def animFunc(self):
                fig = plt.figure()
                sb.heatmap(self.data[0,:,:,0],square = True)
                plt.title(skip + self.filterType + ' Heatmap')
                def animate(i):
                    print('Animating ' + self.filterType + ':' + str(i) + "%",end = "\r")
                    sb.heatmap(self.data[i,:,:,0], cbar = False,square = True)

                anim = animation.FuncAnimation(fig,animate,frames = 100, repeat = False)
                plt.show(block = False)

                writergif = animation.PillowWriter(fps=8) 
                anim.save(self.filterType + skip + casename + '.gif', writer=writergif)

        twoSkipFilter = AnimMeanHeatmap("2_filter",skip_filter2)
        twoSkipFilter.animFunc()
        print("\n")
#         fourSkipFilter = AnimMeanHeatmap("4_filter",skip_filter4)
#         fourSkipFilter.animFunc()
#         print("\n")
#         eightSkipFilter = AnimMeanHeatmap("eight_filter",skip_filter8)
#         eightSkipFilter.animFunc()
#         print("\n")

#saving downscaled data
output2file = h5py.File('outputarray2.h5', 'w')
output2file.create_dataset('dataset_1', data=outputarray_2)
output2file.close()

f = h5py.File("outputarray2.h5",'r')
list(f.keys())
outputarray_2 = f['dataset_1']

#Running Each set of data
# run_case(case = 3, casename = '_3rd_case',skip = 'mean_')
run_case(case = 4,casename = '_4th_case',skip = 'mean_')
# run_case(case = 3, casename = '_3rd_case',skip = 'skip_')
run_case(case = 4,casename = '_4th_case',skip = 'skip_')
# print("Program complete")
