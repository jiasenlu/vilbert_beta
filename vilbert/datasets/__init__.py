from .concept_cap_dataset import ConceptCapLoaderTrain, ConceptCapLoaderVal, ConceptCapLoaderRetrieval
from .vqa_dataset import VQAClassificationDataset
from .refer_expression_dataset import ReferExpressionDataset
from .retreival_dataset import RetreivalDataset, RetreivalDatasetVal
from .vcr_dataset import VCRDataset



# from .flickr_retreival_dataset import FlickrRetreivalDatasetTrain, FlickrRetreivalDatasetVal

__all__ = ["FoilClassificationDataset", \
		   "VQAClassificationDataset", \
		   "ConceptCapLoaderTrain", \
		   "ConceptCapLoaderVal", \
		   "ReferExpressionDataset", \
		   "RetreivalDataset", \
		   "RetreivalDatasetVal",\
		   "VCRDataset", \
		   "ConceptCapLoaderRetrieval"]

DatasetMapTrain = {'TASK0': ConceptCapLoaderTrain,
				   'TASK1': VQAClassificationDataset,
				   'TASK2': VCRDataset,
				   'TASK3': VCRDataset,				   
				   'TASK4': RetreivalDataset,
				   'TASK5': ReferExpressionDataset,
				   }		


DatasetMapEval = {'TASK0': ConceptCapLoaderVal,
				 'TASK1': VQAClassificationDataset,
				 'TASK2': VCRDataset,
				 'TASK3': VCRDataset,				   
				 'TASK4': RetreivalDatasetVal,
				 'TASK5': ReferExpressionDataset,			   
				}
