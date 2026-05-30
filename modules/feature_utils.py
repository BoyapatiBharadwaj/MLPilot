import numpy as np


def get_feature_names(preprocessor):

    feature_names = []

    try:

        transformers = \
            preprocessor.transformers_

        for name, transformer, columns in transformers:

            if name == "num":

                feature_names.extend(columns)

            elif name == "cat":

                encoder = transformer.named_steps[
                    "encoder"
                ]

                try:

                    encoded_names = (
                        encoder
                        .get_feature_names_out(
                            columns
                        )
                        .tolist()
                    )

                    feature_names.extend(
                        encoded_names
                    )

                except Exception:

                    feature_names.extend(
                        columns
                    )

    except Exception:

        pass

    return feature_names