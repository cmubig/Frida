import numpy as np
def get_weight_from_alpha(alpha,epsilon=0.0):
        weight=np.ones(len(alpha))
        # print("weight",weight)
        # print("alpha",alpha)
        for i in range(len(weight)-1):
            temp2=1.0
            for j in range(i+1,len(weight)):
                temp3 = (1.0-alpha[j])
                if temp3 == 0:
                    temp3 = 0.0001
                temp2*=(temp3)
            temp1 = alpha[i]
            weight[i]=temp2*temp1
        weight[-1]=alpha[-1]
        return weight

alpha = np.array([1, 0, 0.54497371, 1, 0.70529471, 0,0,0.49584625])
weight = get_weight_from_alpha(alpha)
print(weight)


def get_alpha_from_weight(weight, epsilon=0.0):
    alpha = np.ones(len(weight))
    alpha[-1] = weight[-1]
    for i in reversed(range(len(weight)-1)):
        temp2 = 1.0
        for j in reversed(range(i+1,len(weight))):
            temp1 = 1-alpha[j]
            if temp1 == 0:
                temp1 = 0.0001
            temp2 *= temp1
        alpha[i] = weight[i]/temp2
    return alpha

alpha_reverse = get_alpha_from_weight(weight, epsilon=0.0)
print(alpha_reverse)
