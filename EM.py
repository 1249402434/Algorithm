class EM:
    def __init__(self,prob):
        #对应于三硬币模型的中各硬币正面出现的概率
        self.pro_a,self.pro_b,self.pro_c=prob

    def e_step(self,Y,size):
        u=[]
        for j in range(size):
            high=self.pro_a*pow(self.pro_b,Y[j])*pow(1-self.pro_b,1-Y[j])
            low=self.pro_a*pow(self.pro_b,Y[j])*pow(1-self.pro_b,1-Y[j])+\
                (1-self.pro_a)*pow(self.pro_c,Y[j])*pow(1-self.pro_c,1-Y[j])
            u.append(high/low)

        return u

    #m_step
    def fit(self,Y,epoch):
        length=len(Y)
        for i in range(1,epoch+1):
            #每一轮开始前先e_step
            result_array = self.e_step(Y, length)
            self.pro_a=1/length*sum(result_array)
            #根据e_step的计算结果计算模型参数的新估计值
            high_b=sum([result_array[j]*Y[j] for j in range(length)])
            low_b=sum(result_array)
            self.pro_b=high_b/low_b

            high_c=sum([(1-result_array[j])*Y[j] for j in range(length)])
            low_c=sum([1-result_array[j] for j in range(length)])
            self.pro_c=high_c/low_c

        return self.pro_a,self.pro_b,self.pro_c

prob=(0.4,0.6,0.7)
em=EM(prob)
Y=[1,1,0,1,0,0,1,0,1,1]
print(em.fit(Y,5))