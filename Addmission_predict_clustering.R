#kMeans Clustering



admissiondata=read.csv("E:/ML/Datasets/Admission_Predict.csv")
head(admissiondata)

str(admissiondata)

x=admissiondata$CGPA
y=admissiondata$Chance.of.Admit

plot(x,y)
admissiondf=data.frame(x,y)

kdata=kmeans(admissiondf,2)
kdata

middle_value=kdata$centers
middle_value
plot(middle_value[,1],middle_value[,2])

kcluster=kdata$cluster
kcluster
plot(admissiondf[,1],admissiondf[,2],col=kcluster)
lines(middle_value[,1],middle_value[,2],type = "p")

kcluster=data.frame(kcluster)
str(kcluster)
kcluster

admissiondf['Clustering']=admissiondf['kcluster']

KClustering=kcluster
cbind(admissiondf,KClustering)

