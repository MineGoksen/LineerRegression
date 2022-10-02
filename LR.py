#!/usr/bin/env python3


from typing import List



class LineerRegression:
    def __init__(self,learning_rate,epoches):
        self.learning_rate = learning_rate;
        self.epoches = epoches;

    def fit(self, X: List[List[int]], y: List[int]):
        
        self.b = 0         
        self.X = X         
        self.y = y                  
        self.yenideger(1, 2, 0, X, y)            

    def predict(self, X: List[List[int]]):
        height = [row[0] for row in X];
        weight = [row[1] for row in X];
        arr = []
        
        for i in range(len(height)):
            arr.append((self.m1*height[i] + self.m2*weight[i]+self.b))
        return arr
  
    def lossfunction(self, m1, m2, b, X: List[List[int]] ,y: List[int]):
        height = [row[0] for row in X];
        weight = [row[1] for row in X];
        z = [];
        
        for i in range(len(height)):
            z.insert(i,(m1*height[i] + m2*weight[i] + b))
         
        summa=0
        for j in range(len(height)):
            error = z[j] - y[j]
            summa += error
        loss = summa/(len(height))    
        return loss
    
    def lossfunction1(self, m1, m2, b, X: List[List[int]] ,y: List[int]):
        height = [row[0] for row in X];
        weight = [row[1] for row in X];
        z = [];
        
        for i in range(len(height)):
            z.insert(i,(m1*height[i] + m2*weight[i] + b))
         
        summa=0
        for j in range(len(height)):
            error = (z[j] - y[j]) * height[j]
            summa += error
        loss = summa/(len(height))    
        return loss
    
    def lossfunction2(self, m1, m2, b, X: List[List[int]] ,y: List[int]):
        height = [row[0] for row in X];
        weight = [row[1] for row in X];
        z = [];
        
        for i in range(len(height)):
            z.insert(i,(m1*height[i] + m2*weight[i] + b))
         
        summa=0
        for j in range(len(height)):
            error = (z[j] - y[j])*weight[j]
            summa += error
        loss = summa/(len(height))    
        return loss

    def yenideger(self,m1,m2,b,X: List[List[int]] ,y: List[int]):
        
        height = [row[0] for row in X];
        weight = [row[1] for row in X];
        
        i=0
        for i in range((self.epoches)):
            loss2 = self.lossfunction(m1,m2,b,X,y)
            lossm1 = self.lossfunction1(m1,m2,b,X,y)
            lossm2 = self.lossfunction2(m1,m2,b,X,y)
            m1 = m1- self.learning_rate* (2*lossm1)
            m2 = m2- self.learning_rate* (2*lossm2)
            b = b- self.learning_rate* (2*loss2)
        
            self.m1 = (m1)
            self.m2 = (m2)
            self.b = (b)
        return (m1),(m2),(b)   


    

  