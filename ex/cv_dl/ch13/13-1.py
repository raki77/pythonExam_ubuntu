import numpy as np

X=np.array([[169,70],[172,68],[175,78],[163,58],[180,80],[159,76],[158,52],[173,69],[180,75],[155,50],[187,90],[170,66]])

m=np.mean(X,axis=0)					# 모델 학습
cv=np.cov(X,rowvar=False)

gen=np.random.multivariate_normal(m,cv,5)	# 샘플 생성

print(gen)