import numpy as np



class BoundaryAttack():
    def __init__(self, model, epsilon = 0.9, delta = 0.1, threshold = 1e-1, seed = 42, backend = 'lvq'):
        self.model = model
        self.epsilon = epsilon
        self.delta = delta
        self.t_s = threshold
        self.seed = seed
        self.backend = backend

	
    # def get_diff(self, sample_1, sample_2):
	#     return np.linalg.norm(sample_1 - sample_2)
	


    def initialize(self,unit, data, labels, correct_label):
        rng = np.random.default_rng(self.seed)   
        if self.attack_label == None:
            wanted_indices = np.flatnonzero(labels != correct_label)
            wanted_data = data[wanted_indices]
            dist = [self.get_diff(unit, wanted_data[i]) for i in range(len(wanted_data))]
            sorted_indices = np.argsort(np.array(dist))
            sorted_data = sorted_indices
            for i in range(len(sorted_data)):
                if self.model.predict(sorted_data[i]) != correct_label:
                    index= i
                    break
 
        else:
            index = rng.choice(np.flatnonzero(labels == self.attack_label))

        return data[index]       
    
	
    def has_nan(self, arr):
        for element in arr:
            if np.isnan(element):
                return True
        return False


	


    def forward_perturbation(self, prev_sample, target_sample):
        perturbed = (target_sample - prev_sample).astype(np.float32)
        perturbed *= self.epsilon
        return perturbed



    def orthogonal_perturbation(self, data, prev_sample, target_sample):
        """Generate orthogonal perturbation."""
        rng = np.random.default_rng(self.seed)  
        perturb = rng.uniform(low=0, high=1, size=(1, data.shape[1]))
        perturb /= np.linalg.norm(perturb)
        perturb *= self.delta * np.mean(self.get_diff(target_sample, prev_sample))

        # Project perturbation onto sphere around target
        diff = (target_sample - prev_sample).astype(np.float32)

        # Check for potential division by zero
        norm_diff = np.linalg.norm(diff)
        if norm_diff < 1e-8:
            return np.zeros_like(perturb)  # Avoid division by nearly zero

        # Orthogonal unit vector
        diff /= norm_diff

        # Project onto the orthogonal then subtract from perturb
        # to get projection onto the sphere surface
        perturb -= (np.vdot(perturb, diff) / np.linalg.norm(diff)**2 + 1e-8) * diff

        # Check overflow and underflow
        overflow = (prev_sample + perturb) - data.max() + data.mean(axis=0)
        perturb -= overflow * (overflow > 0)
        underflow = -1 * (data.mean(axis=0))
        perturb += underflow * (underflow > 0)

        return perturb

    def fit(self, data, labels):
        self.data = data
        self.labels = labels






    def boundary_attack(self, target_sample, attack_type = 'untargeted', targeted_class = None):
        if self.backend == 'lvq':
            if attack_type == 'untargeted':
                self.attack_label = None
            else:
                self.attack_label = targeted_class
                
            if attack_type == 'tagerted' and targeted_class == None:
                raise ValueError('Specify Class')


            correct_label = self.model.predict(target_sample)

            initial_sample = self.initialize(target_sample, self.data, self.labels, correct_label)
            attack_class = self.model.predict(initial_sample)
            adversarial_sample = initial_sample
            n_steps = 0
            n_calls = 0

            # Move first step to the boundary
            while True:
                trial_sample = adversarial_sample + self.forward_perturbation(adversarial_sample, target_sample)
                prediction = self.model.predict(np.array(trial_sample))
                n_calls += 1
                if prediction == attack_class:
                    adversarial_sample = trial_sample
                    break
                else:
                    self.epsilon *= 0.9

            # Iteratively run attack
            while True:
                print("Step #{}...".format(n_steps))
                # Orthogonal step
                print("\tDelta step...")
                d_step = 0
                while True:
                    d_step += 1
                    print("\t#{}".format(d_step))
                    trial_samples = []
                    for i in np.arange(20):
                        trial_sample = adversarial_sample + self.orthogonal_perturbation(self.data, adversarial_sample, target_sample)[0]
                        trial_samples.append(trial_sample)
                
                    predictions1 = []
                    for sample in trial_samples:
                        score = self.model.predict(sample)
                        predictions1.append(score)

                    predictions = np.array(predictions1).flatten()
                    n_calls += 10
                    print(len(predictions))

                    d_score = np.mean(predictions == attack_class)
                    print(d_score)
                    if d_step %40 == 0:
                        break



                    if d_score > 0.0:
                        if d_score < 0.3:
                            self.delta *= 0.9
                        elif d_score > 0.7:
                            self.delta /= 0.9
                        adversarial_sample = np.array(trial_samples)[np.flatnonzero(predictions == attack_class)][0]
                        break
                    elif d_score == 0.0:
                        self.delta *= 0.9

                # Forward step
                print("\tEpsilon step...")
                e_step = 0
                while True:
                    e_step += 1
                    print("\t#{}".format(e_step))
                    trial_sample = adversarial_sample + self.forward_perturbation(adversarial_sample, target_sample)
                    prediction = self.model.predict(trial_sample)
                    n_calls += 1
                    if prediction == attack_class:
                        adversarial_sample = trial_sample
                        self.epsilon /= 0.5
                        break
                    elif e_step > 500:
                            break
                    else:
                        self.epsilon *= 0.5
                n_steps += 1
                chkpts = [1, 5, 10, 50, 100, 500]
                if (n_steps in chkpts) or (n_steps % 50 == 0):
                    print("{} steps".format(n_steps))
                    print("{} diff".format(self.get_diff(adversarial_sample, target_sample)))
                diff = np.mean(self.get_diff(adversarial_sample, target_sample))
                if diff <= self.t_s or e_step > 500 or (n_steps % 500 == 0):
                    print("{} steps".format(n_steps))
                    print("Mean Squared Error: {}".format(diff))
                    if self.model.predict(adversarial_sample) == correct_label:
                    
                        raise ValueError(f'Unsuccesful, actual_class: {correct_label} same as adv_class: {self.model.predict(adversarial_sample)}')
                
                    return adversarial_sample, diff, self.model.predict(adversarial_sample)
                



        elif self.backend == 'sklearn':
            import pandas as pd
            if isinstance(self.data, pd.DataFrame) == False:
                raise TypeError('Not Dataframe, might want to retrain with dataframe')
            if attack_type == 'untargeted':
                self.attack_label = None
            else:
                self.attack_label = targeted_class
                
            if attack_type == 'tagerted' and targeted_class == None:
                raise ValueError('Specify Class')


            correct_label = self.model.predict(pd.DataFrame(np.array(target_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))

            initial_sample = self.initialize(target_sample, self.data, self.labels, correct_label)
            attack_class = self.model.predict(pd.DataFrame(np.array(initial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))
            adversarial_sample = initial_sample
            n_steps = 0
            n_calls = 0

            # Move first step to the boundary
            while True:
                trial_sample = adversarial_sample + self.forward_perturbation(adversarial_sample, target_sample)
                prediction = self.model.predict(pd.DataFrame(np.array(trial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))
                n_calls += 1
                if prediction == attack_class:
                    adversarial_sample = trial_sample
                    break
                else:
                    self.epsilon *= 0.9

            # Iteratively run attack
            while True:
                print("Step #{}...".format(n_steps))
                # Orthogonal step
                print("\tDelta step...")
                d_step = 0
                while True:
                    d_step += 1
                    print("\t#{}".format(d_step))
                    trial_samples = []
                    for i in np.arange(20):
                        trial_sample = adversarial_sample + self.orthogonal_perturbation(self.data, adversarial_sample, target_sample)[0]
                        trial_samples.append(trial_sample)
                
                    predictions1 = []
                    for sample in trial_samples:
                        score = self.model.predict(pd.DataFrame(np.array(sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))
                        predictions1.append(score)

                    predictions = np.array(predictions1).flatten()
                    n_calls += 10
                    print(len(predictions))

                    d_score = np.mean(predictions == attack_class)
                    print(d_score)
                    if d_step %40 == 0:
                        break



                    if d_score > 0.0:
                        if d_score < 0.3:
                            self.delta *= 0.9
                        elif d_score > 0.7:
                            self.delta /= 0.9
                        adversarial_sample = np.array(trial_samples)[np.flatnonzero(predictions == attack_class)][0]
                        break
                    elif d_score == 0.0:
                        self.delta *= 0.9

                # Forward step
                print("\tEpsilon step...")
                e_step = 0
                while True:
                    e_step += 1
                    print("\t#{}".format(e_step))
                    trial_sample = adversarial_sample + self.forward_perturbation(adversarial_sample, target_sample)
                    prediction = self.model.predict(pd.DataFrame(np.array(sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))
                    n_calls += 1
                    if prediction == attack_class:
                        adversarial_sample = trial_sample
                        self.epsilon /= 0.5
                        break
                    elif e_step > 500:
                            break
                    else:
                        self.epsilon *= 0.5
                n_steps += 1
                chkpts = [1, 5, 10, 50, 100, 500]
                if (n_steps in chkpts) or (n_steps % 50 == 0):
                    print("{} steps".format(n_steps))
                    print("{} diff".format(self.get_diff(adversarial_sample, target_sample)))
                diff = np.mean(self.get_diff(adversarial_sample, target_sample))
                if diff <= self.t_s or e_step > 500 or (n_steps % 500 == 0):
                    print("{} steps".format(n_steps))
                    print("Mean Squared Error: {}".format(diff))
                    if self.model.predict(pd.DataFrame(np.array(adversarial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns)) == correct_label:
                    
                        raise ValueError(f'Unsuccesful, actual_class: {correct_label} same as adv_class: {self.model.predict(pd.DataFrame(np.array(adversarial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))}')
                
                    return adversarial_sample, diff, self.model.predict(pd.DataFrame(np.array(adversarial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns))
        
        
        
        elif self.backend == 'TF2':
            import pandas as pd
            if isinstance(self.data, pd.DataFrame) == False:
                raise TypeError('Not Dataframe, might want to retrain with dataframe')
            if attack_type == 'untargeted':
                self.attack_label = None
            else:
                self.attack_label = targeted_class
                
            if attack_type == 'tagerted' and targeted_class == None:
                raise ValueError('Specify Class')


            correct_label = np.argmax(self.model.predict(pd.DataFrame(np.array(target_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])

            initial_sample = self.initialize(target_sample, self.data, self.labels, correct_label)
            attack_class = np.argmax(self.model.predict(pd.DataFrame(np.array(initial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])
            adversarial_sample = initial_sample
            n_steps = 0
            n_calls = 0

            # Move first step to the boundary
            while True:
                trial_sample = adversarial_sample + self.forward_perturbation(adversarial_sample, target_sample)
                prediction = np.argmax(self.model.predict(pd.DataFrame(np.array(trial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])
                n_calls += 1
                if prediction == attack_class:
                    adversarial_sample = trial_sample
                    break
                else:
                    self.epsilon *= 0.9

            # Iteratively run attack
            while True:
                print("Step #{}...".format(n_steps))
                # Orthogonal step
                print("\tDelta step...")
                d_step = 0
                while True:
                    d_step += 1
                    print("\t#{}".format(d_step))
                    trial_samples = []
                    for i in np.arange(20):
                        trial_sample = adversarial_sample + self.orthogonal_perturbation(self.data, adversarial_sample, target_sample)[0]
                        trial_samples.append(trial_sample)
                
                    predictions1 = []
                    for sample in trial_samples:
                        score = np.argmax(self.model.predict(pd.DataFrame(np.array(sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])
                        predictions1.append(score)

                    predictions = np.array(predictions1).flatten()
                    n_calls += 10
                    print(len(predictions))

                    d_score = np.mean(predictions == attack_class)
                    print(d_score)
                    if d_step %40 == 0:
                        break



                    if d_score > 0.0:
                        if d_score < 0.3:
                            self.delta *= 0.9
                        elif d_score > 0.7:
                            self.delta /= 0.9
                        adversarial_sample = np.array(trial_samples)[np.flatnonzero(predictions == attack_class)][0]
                        break
                    elif d_score == 0.0:
                        self.delta *= 0.9

                # Forward step
                print("\tEpsilon step...")
                e_step = 0
                while True:
                    e_step += 1
                    print("\t#{}".format(e_step))
                    trial_sample = adversarial_sample + self.forward_perturbation(adversarial_sample, target_sample)
                    prediction = np.argmax(self.model.predict(pd.DataFrame(np.array(sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])
                    n_calls += 1
                    if prediction == attack_class:
                        adversarial_sample = trial_sample
                        self.epsilon /= 0.5
                        break
                    elif e_step > 500:
                            break
                    else:
                        self.epsilon *= 0.5
                n_steps += 1
                chkpts = [1, 5, 10, 50, 100, 500]
                if (n_steps in chkpts) or (n_steps % 50 == 0):
                    print("{} steps".format(n_steps))
                    print("{} diff".format(self.get_diff(adversarial_sample, target_sample)))
                diff = np.mean(self.get_diff(adversarial_sample, target_sample))
                if diff <= self.t_s or e_step > 500 or (n_steps % 500 == 0):
                    print("{} steps".format(n_steps))
                    print("Mean Squared Error: {}".format(diff))
                    if np.argmax(self.model.predict(pd.DataFrame(np.array(adversarial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0]) == correct_label:
                    
                        raise ValueError(f'Unsuccesful, actual_class: {correct_label} same as adv_class: {np.argmax(self.model.predict(pd.DataFrame(np.array(adversarial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])}')
                
                    return adversarial_sample, diff, np.argmax(self.model.predict(pd.DataFrame(np.array(adversarial_sample).reshape((1, self.data.shape[1])), columns  = self.data.columns).iloc[0:1])[0])


    def get_diff(self, sample_1, sample_2):
	    return np.linalg.norm(sample_1 - sample_2)
    








