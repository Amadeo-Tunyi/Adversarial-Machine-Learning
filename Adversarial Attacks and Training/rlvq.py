import numpy as np
class RLVQ:
    def __init__(self, num_prototypes_per_class, initialization_type = 'mean', learning_rate = 0.05,max_iter = 100, test_data = None, test_labels = None):

        self.max_iter = max_iter 
        self.test_data = test_data
        self.test_labels = test_labels
        self.num_prototypes = num_prototypes_per_class
        self.initialization_type = initialization_type
        self.alpha = learning_rate
    
    
    

    def initialization(self, train_data, train_labels):
        if self.initialization_type == 'mean':
            """Prototype initialization: if number of prototypes is 1, prototype initialised is the mean
            if prototype is n>1, prototype initilised is the mean plus n-1 points closest to mean"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            protolabels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)
                    list1.append(mu)#.astype(int))
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
            
                P = np.array(prototypes) 
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    class_data = train_data[index]
                    mu = np.mean(class_data, axis = 0)

                    distances = [(mu-c)@(mu-c).T for c in class_data]
                    index = np.argsort(distances)
                    indices = index[1:self.num_prototypes]
                    prototype = class_data[indices]
                    r = np.vstack((mu, prototype))
                    list2.append(r)
                    ind = []
                    for j in range(self.num_prototypes ):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)#.flatten()   
                prototypes = M.reshape(num_protos,num_dims)
               
                P = np.array(prototypes)
            return np.array(new_labels).flatten(), P
        
        elif self.initialization_type == 'random':
            """Prototype initialization random: randomly chooses n points per class"""
            num_dims = train_data.shape[1]
            labels = train_labels.astype(int)
            #self.train_data = self.normalize(self.train_data)
        
        
            unique_labels = np.unique(labels)

            num_protos = self.num_prototypes * len(unique_labels)

            protolabels =  unique_labels
            new_labels = []
            list1 = []
            if self.num_prototypes == 1:
                for i in unique_labels:
                    index = np.flatnonzero(labels == i)
                    random_int = np.random.choice(np.array(index))
                    prototype = train_data[random_int]
                    list1.append(prototype)
                prototypes = np.array(list1).reshape(len(unique_labels),num_dims)
                #regulate the prototypes, could also be done with GMM
                P = np.array(prototypes) 
                new_labels = unique_labels
            else:
                list2 = []
                for i in unique_labels:
            
                    index = np.flatnonzero(labels == i)
                    random_integers = np.random.choice(np.array(index), size=self.num_prototypes)
                    prototype = train_data[random_integers]
                    list2.append(prototype)
                    ind = []
                    for j in range(self.num_prototypes):
                        ind.append(i)
                        
                    new_labels.append(ind) 
                    M = np.array(list2)  
                prototypes = M.reshape(num_protos,num_dims)
                P = np.array(prototypes) 
            return np.array(new_labels).flatten(), P 
    

    def weights(self, data):
        weight = np.full(data.shape[1], fill_value = 1/data.shape[1])
        
        return weight
    

    def dist(self, x, y, w):
    
        r =  [(w[i]*(x[i] -y[i]))**2 for i in range(len(x))]
        f = np.sqrt(np.array(r).sum())
        return f
    

    def weight_update(self, weight,data, label, prototypes,  protolabels, eps):

        beta = 1e-8
        
        for i in range(len(data)):
            xi = data[i]
            xlabel = label[i]
            distances = np.array([self.dist(xi, p, weight) for p in prototypes])
            nearest_index = np.argmin(distances)
            if xlabel == protolabels[nearest_index]:
                weight -= eps*(weight*(np.subtract(xi, prototypes[nearest_index]))**2)
            else:
                weight += eps*(weight*(np.subtract(xi, prototypes[nearest_index]))**2)
            weight = weight.clip(min = 0)
            weight = weight/weight.sum()
        return weight       


    def proto_update(self, data, label,weight, protolabels, prototypes, alpha):
    
        for i in range(len(data)):
            xi = data[i]
            xlabel = label[i]

            distances = np.array([self.dist(xi, p, weight) for p in prototypes])
            nearest_index = np.argmin(distances)

            
            if xlabel == protolabels[nearest_index]:
                prototypes[nearest_index] += alpha*(xi - prototypes[nearest_index])
            else:
                prototypes[nearest_index] -=  alpha*(xi - prototypes[nearest_index])
                    

        return prototypes
    


    def fit(self, data, labels,eps_zero = 0.1, alpha_zero = 0.1, max_iter = 100, decay_scheme = True):
        import math
        self.protolabels, self.prototypes = self.initialization(data, labels)
        self.weight = self.weights(data)
        iter = 0
        while iter < max_iter:
            if decay_scheme == True:
                eps = eps_zero*math.exp(-1*iter/max_iter)
                alpha = alpha_zero*math.exp(-1*iter/max_iter)
                self.weight = self.weight_update(self.weight, data, labels, self.prototypes, self.protolabels, eps)
                self.prototypes = self.proto_update(data, labels, self.weight, self.protolabels, self.prototypes, alpha)

            else:
                
                self.weight = self.weight_update(self.weight, data, labels, self.prototypes, self.protolabels, eps_zero)
                self.prototypes = self.proto_update(data, labels, self.weight, self.protolabels, self.prototypes, alpha_zero)
            iter += 1
        return self.prototypes, self.protolabels, self.weight


    def predict_all(self, data, return_scores = False):

        """predict an array of instances""" 
        label = []
        #prototypes, _ = RSLVQ(data, labels, num_prototypes, max_iter)
        if return_scores == False:
            for i in range(data.shape[0]):
                xi = data[i]
                distances = np.array([np.linalg.norm(xi - p) for p in self.prototypes])
                index = np.argwhere(distances == distances.min())
                x_label = self.protolabels[index]
                label.append(x_label)
            return np.array(label).flatten()
        else:
            predicted = []
            for i in range(len(data)):
                predicted.append(self.proba_predict(data[i]))
            return predicted 
    


    def evaluate(self, test_data, test_labels):
        """predict over test set and outputs test MAE"""
        predicted = []
        for i in range(len(test_data)):
            predicted.append(self.predict(test_data[i]))
        val_acc = (np.array(predicted) == np.array(test_labels).flatten()).mean() * 100 
        return val_acc
    


    def predict(self, input):
        """predicts only one output at the time, numpy arrays only, 
        might want to convert"""
        
   


       
         
   
        distances = np.array([np.linalg.norm(input - p) for p in self.prototypes])
        index = np.argmin(distances)
        x_label = self.protolabels[index]
        
        return x_label
    



    def proba_predict(self, input):
        """probabilistic prediction of a point by approximation of distances of a point to closest prototypes
        the argmin is the desired class"""
        scores = []
        closest_prototypes = []
        for i in np.unique(self.protolabels):
            label_prototypes = self.prototypes[np.flatnonzero(self.protolabels == i)]
            distances = np.array([np.linalg.norm(input - label_prototypes[j]) for j in range(label_prototypes.shape[0])])
            closest_prototype = label_prototypes[np.argmin(distances)]
            closest_prototypes.append(closest_prototype)
        dists = np.array([np.linalg.norm(input - prototype) for prototype in closest_prototypes])
        scores = np.array([d/dists.sum() for d in dists])
        return scores 