from drug_generator.config import config
from keras.layers import TextVectorization
from keras.utils import to_categorical


class preprocess_data:
    def fit(self,X=None,y=None):
        self.input_text_vectorization_layer = TextVectorization(
            max_tokens=len(config.BASE_VOCABULARY)+2,
            standardize=None,
            split="character",
            output_sequence_length=config.MAX_INPUT_SEQUENCE_LEN,
            vocabulary=config.BASE_VOCABULARY
        )

        self.output_text_vectorization_layer = TextVectorization(
            max_tokens=len(config.BASE_VOCABULARY)+2,
            standardize=None,
            split="character",
            output_sequence_length=config.MAX_OUTPUT_SEQUENCE_LEN,
            vocabulary=config.BASE_VOCABULARY
        )
        return self

    def transform(self,X=None,y=None):
        self.X = self.input_text_vectorization_layer(X).numpy()
        self.Y = self.output_text_vectorization_layer(y).numpy()
        return self.X,self.Y
