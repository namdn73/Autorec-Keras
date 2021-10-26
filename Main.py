import Data_Processing
import Autorec

train, test = Data_Processing.data_processing(path=r".\ml-1m\ratings.dat", dat_file=True)
print(train.shape)
autorec = Autorec.Autorec()
autorec.Model(train, 600)
autorec.fit(train, y_val=test, epoch=1)