import pandas as pd
import sys
import numpy as nm
from collections import defaultdict
import operator

output_file = open("chawla_jitesh_clustering.txt",'w')

#  Initializations
if __name__ == '__main__':
    inputfile = str(sys.argv[1])
    initialPoints = str(sys.argv[2])
    K = int(sys.argv[3])

    iterations = int(sys.argv[4])
pd.set_option('max_columns', 50)
pd.set_option('display.width', 1000)
column_list = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety','Class']
rand_index_list=[]
centroid_list=[]
centroid_list_final=[]
min_dist_list_index=[]
sum2=[]
enum_final=[]

row_list = []

initial_cluster_dict= {}
temp_dict={'unacc':0,'acc':0,'good':0,'vgood':0}

suml12 = 0

intermediate_cluster_dict= defaultdict(list)



utility_matrix1 = pd.read_csv(inputfile, header=None, names=column_list)
utility_matrix1_raw = pd.read_csv(inputfile, header=None, names=column_list)
utility_matrix2 = pd.read_csv(initialPoints, header=None, names=column_list)
utility_matrix1=utility_matrix1.drop('Class',axis=1)
utility_matrix2=utility_matrix2.drop('Class',axis=1)

#  Replacing the the values of categorical variables with integers.
def replace(utility_matrix1):

    for j in range(len(utility_matrix1)):
        if utility_matrix1['doors'][j]==int(2):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['doors'][j], int(1))
        elif utility_matrix1['doors'][j] == int(3):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['doors'][j], int(2))
        elif utility_matrix1['doors'][j] == int(4):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['doors'][j], int(3))
        elif utility_matrix1['doors'][j] == str('5more'):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['doors'][j], int(4))
        if utility_matrix1['persons'][j] == int(2):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['persons'][j], int(1))
        elif utility_matrix1['persons'][j] == int(4):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['persons'][j], int(2))
        if utility_matrix1['persons'][j] == str('more'):
            utility_matrix1 = utility_matrix1.replace(utility_matrix1['persons'][j], int(3))

    utility_matrix1 =utility_matrix1.replace('vhigh', int(4))
    utility_matrix1= utility_matrix1.replace('high', int(3))
    utility_matrix1 = utility_matrix1.replace('small', int(1))
    utility_matrix1= utility_matrix1.replace('low', int(1))
    utility_matrix1 = utility_matrix1.replace('med', int(2))
    utility_matrix1 = utility_matrix1.replace('big', int(3))
    utility_matrix = utility_matrix1.loc[:,:].applymap(float)

    utility_matrix1_list = []

    for i in range(len(utility_matrix1)):

        utility_matrix1_list.append(list(utility_matrix1.loc[i][:]))

    # print(utility_matrix1_list)

    for n in range(len(utility_matrix1_list)):
        for j in range(len(utility_matrix1_list[n])):
            utility_matrix1_list[n][j] = int(utility_matrix1_list[n][j])
    return utility_matrix,utility_matrix1_list

um,utility_matrix1_list=replace(utility_matrix1)
cm,centroid_list_final=replace(utility_matrix2)
# print(cm)
for key in range(K):
    initial_cluster_dict[key]= centroid_list_final[key]

for i in utility_matrix1_list:
    dist_list=[]
    # min_dist_list_index=[]
    for k in centroid_list_final:
        sum1 = 0
        dist = 0
        for z in range(0,len(i)):
            sum1+= (i[z]-k[z])**2

        dist=nm.sqrt(sum1)

        dist_list.append(dist)

    # print(dist_list)

    min_dist_list_index.append(dist_list.index(min(dist_list)))

for i in list(initial_cluster_dict.keys()):

    for ele in range(len(min_dist_list_index)):

        if i == min_dist_list_index[ele]:

            intermediate_cluster_dict[i].append(ele)

# print(dict(intermediate_cluster_dict))


# Finding the new centroids
new_centroids=[]

for k in range(K):

    new_centroids.append(list(um.loc[intermediate_cluster_dict[k]].mean(axis=0)))

count_len=0


# for key in range(K):
for value in dict(intermediate_cluster_dict)[key]:

        count_len+=1
#  Giving cluster names and categgorizing.
def cluster_name(cluster_kmeans_dict,um):
    # temp_dict={}
    suml12=0
    for clus_key in cluster_kmeans_dict.keys():
        for j in cluster_kmeans_dict[clus_key]:
            if (list(um.loc[j])== utility_matrix1_list[j]):
                # print(utility_matrix1_raw['Class'][j])
                temp_dict[utility_matrix1_raw['Class'][j]] += 1
                # temp_dict['unacc'] += 1
        for key, value in temp_dict.items():
            if (key != max(temp_dict.items(), key=operator.itemgetter(1))[0]):
                suml12 += temp_dict[key]
        print("cluster: " + str(max(temp_dict.items(), key=operator.itemgetter(1))[0]))
        output_file.write("cluster: " + max(temp_dict.items(), key=operator.itemgetter(1))[0]+ "\n")
        for j in cluster_kmeans_dict[clus_key]:
            print(list(utility_matrix1_raw.loc[j]))
            output_file.write(str(list(utility_matrix1_raw.loc[j]))+ "\n")
        output_file.write("\n\n")
        print("\n")

    print("Number of points wrongly assigned:")
    output_file.write("Number of points wrongly assigned:")
    print(suml12)
    output_file.write("\n"+str(suml12))

#  Finding Kmeans
def k_means(utility_matrix1_list,new_centroids,iterations):

    cluster_kmeans_dict={}

    for i in range(K):
        cluster_kmeans_dict[i]=[]
    temp=new_centroids

    for i in utility_matrix1_list:

        dist_list = []


        for coordinates in new_centroids:

            sum1 = 0

            dist = 0


            for z in range(0, len(i)):

                sum1 += (i[z] - coordinates[z]) ** 2

            dist = nm.sqrt(sum1)

            dist_list.append(dist)

        ind=dist_list.index(min(dist_list))

        cluster_kmeans_dict[ind].append(utility_matrix1_list.index(i))

    # print (dict(cluster_kmeans_dict))

    new_centroids = []

    for k in range(K):
        new_centroids.append(list(um.loc[dict(cluster_kmeans_dict)[k]].mean(axis=0)))

    #print new_centroids

    if iterations==0 :

        # print(cluster_kmeans_dict)
        cluster_name(cluster_kmeans_dict, um)

        return

    iterations-=1

    k_means(utility_matrix1_list,new_centroids,iterations)

k_means(utility_matrix1_list,new_centroids,iterations)
