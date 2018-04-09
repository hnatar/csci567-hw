import numpy as np
from kmeans import KMeans


class GMM():
    '''
        Fits a Gausian Mixture model to the data.

        attrs:
            n_cluster : Number of mixtures
            e : error tolerance
            max_iter : maximum number of updates
            init : initialization of means and variance
                Can be 'random' or 'kmeans'
            means : means of gaussian mixtures
            variances : variance of gaussian mixtures
            pi_k : mixture probabilities of different component
    '''

    def __init__(self, n_cluster, init='k_means', max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.e = e
        self.max_iter = max_iter
        self.init = init
        self.means = None
        self.variances = None
        self.pi_k = None

    def fit(self, x):
        '''
            Fits a GMM to x.

            x: is a NXD size numpy array
            updates:
                self.means
                self.variances
                self.pi_k
        '''
        assert len(x.shape) == 2, 'x can only be 2 dimensional'

        np.random.seed(42)
        N, D = x.shape

        if (self.init == 'k_means'):
            # TODO
            # - comment/remove the exception
            # - initialize means using k-means clustering
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            """ Get mu_k-s and membership from k-Means """
            means, kmeans_R, _ = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e).fit(x)
            self.means = means
            """ Count points in each cluster, set pi_k using that """
            self.pi_k = np.zeros(self.n_cluster)
            for i in range(0, self.n_cluster):
                self.pi_k[i] = float( len(kmeans_R[kmeans_R==i]) ) / N
            """ Initialize covariance """
            cov = np.zeros(self.n_cluster * D * D).reshape( (self.n_cluster, D, D) )
            for k in range(0, self.n_cluster):
                N_k = float( len(kmeans_R[kmeans_R==k]) )
                for n in range(0, N):
                    if kmeans_R[n] == k:
                        Y = (x[n]-self.means[k]).reshape(D,1)
                        R1 = np.matmul( Y, Y.transpose() )
                        cov[k] += R1
                cov[k] = cov[k]/N_k
            self.variances = cov
            # DONOT MODIFY CODE BELOW THIS LINE

        elif (self.init == 'random'):
            # TODO
            # - comment/remove the exception
            # - initialize means randomly
            # - compute variance and pi_k

            # DONOT MODIFY CODE ABOVE THIS LINE
            means = np.zeros(self.n_cluster * D).reshape( (self.n_cluster, D) )
            """ Sample each dimension for mu_k-s uniformly from [0,1] """
            for i in range(0, self.n_cluster):
                means[i] = np.random.rand( D )
            self.means = means
            """ Variance and priors """
            self.variances = np.zeros(self.n_cluster * D * D).reshape( (self.n_cluster, D, D) )
            for k in range(0, self.n_cluster):
                self.variances[k] = np.identity( D )
            self.pi_k = np.array( [1.0/self.n_cluster]*self.n_cluster )
            print("Random Sampling:")
            print("MEANS = ", self.means)
            for k in range(0, self.n_cluster):
                print("COV = ", self.variances[k])
                print("PRIOR = ", self.pi_k[k])
            # DONOT MODIFY CODE BELOW THIS LINE

        else:
            raise Exception('Invalid initialization provided')

        # TODO
        # - comment/remove the exception
        # - find the optimal means, variances, and pi_k and assign it to self
        # - return number of updates done to reach the optimal values.
        # Hint: Try to seperate E & M step for clarity

        # DONOT MODIFY CODE ABOVE THIS LINE
        def multivariate_gaussian_pdf(x, mean, cov):
            D = x.shape[0]
            while np.linalg.matrix_rank(cov) < D:
                cov += 0.001 * np.identity(D)
            cov_rank = np.linalg.matrix_rank(cov)
            cov_inv = np.linalg.inv(cov)
            exp_value = -0.5 * np.matmul((x-mean).transpose(), np.matmul(cov_inv,x-mean))
            return np.exp(exp_value)/(np.power(2*np.pi, 0.5*D)*np.sqrt(np.linalg.det(cov_inv)))
        """ Compute the log-likelihood """
        loglike = self.compute_log_likelihood(x)
        Gamma = np.zeros(N*self.n_cluster).reshape( (N, self.n_cluster) )
        
        i = 0   
        while i<self.max_iter:
            """ E-step """

            for n in range(0, N):
                """ Probability of n-th sample coming from k-th model """
                gamma_nk = np.array( [0.0]*self.n_cluster )
                for k in range(0, self.n_cluster):
                    gamma_nk[k] += self.pi_k[k] * multivariate_gaussian_pdf(x[n], self.means[k], self.variances[k])
                gamma_nk = gamma_nk / np.sum(gamma_nk)
                Gamma[n] = gamma_nk
                assert abs(np.sum(Gamma[n]) - 1) < self.e
            assert abs(np.sum(Gamma) - N) < self.e
            """ M-step """
            N_k = np.sum( Gamma, axis=0 )
            for k in range(0, self.n_cluster):
                """ Update for mean """
                self.means[k] = np.zeros(D)
                for n in range(0, N):
                    self.means[k] += (Gamma[n][k]*x[n])/N_k[k]
                """ Update for covariance """
                self.variances[k] = np.zeros(D*D).reshape( (D,D) )
                for n in range(0, N):
                    temp = (x[n]-self.means[k]).reshape(D,1)
                    self.variances[k] += Gamma[n][k]*np.matmul(temp, temp.transpose()) / N_k[k]
            """ Update for pi_k """
            self.pi_k = N_k/N
            """ Compute updated log-likelihood """
            newll = self.compute_log_likelihood(x)
            if abs(newll - loglike) < self.e:
                break
            loglike = newll
            i+=1
        return i
        # DONOT MODIFY CODE BELOW THIS LINE

    def sample(self, N):
        '''
        sample from the GMM model

        N is a positive integer
        return : NXD array of samples

        '''
        assert type(N) == int and N > 0, 'N should be a positive integer'
        np.random.seed(42)
        if (self.means is None):
            raise Exception('Train GMM before sampling')

        # TODO
        # - comment/remove the exception
        # - generate samples from the GMM
        # - return the samples

        # DONOT MODIFY CODE ABOVE THIS LINE
        # DONOT MODIFY CODE BELOW THIS LINE

    def compute_log_likelihood(self, x):
        '''
            Return log-likelihood for the data

            x is a NXD matrix
            return : a float number which is the log-likelihood of data
        '''
        assert len(x.shape) == 2,  'x can only be 2 dimensional'
        # TODO
        # - comment/remove the exception
        # - calculate log-likelihood using means, variances and pi_k attr in self
        # - return the log-likelihood
        # Note: you can call this function in fit function (if required)
        # DONOT MODIFY CODE ABOVE THIS LINE
        def multivariate_gaussian_pdf(x, mean, cov):
            D = x.shape[0]
            while np.linalg.matrix_rank(cov) < D:
                cov += 0.001 * np.identity(D)
            cov_rank = np.linalg.matrix_rank(cov)
            cov_inv = np.linalg.inv(cov)
            exp_value = -0.5 * np.matmul((x-mean).transpose(), np.matmul(cov_inv,x-mean))
            return np.exp(exp_value)/(np.power(2*np.pi, 0.5*D)*np.sqrt(np.linalg.det(cov_inv)))
        """ Compute log likelihood """
        N, D = x.shape
        loglike = 0.0
        for n in range(0, N):
            px = 0.0
            for z in range(0, self.n_cluster):
                px += self.pi_k[z] * multivariate_gaussian_pdf(x[n], self.means[z], self.variances[z])
            loglike += np.log(px)
        return float(loglike)
        # DONOT MODIFY CODE BELOW THIS LINE
