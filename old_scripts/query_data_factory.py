import os
print(os.getcwd())
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from data_factory import DataFactory

factory = DataFactory()
cs = factory.get_dev('pubmed')
