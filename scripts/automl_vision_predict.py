# import argparse
#
# parser = argparse.ArgumentParser(description='automl vision predict')
# parser.add_argument('project_id', help='project id')
# parser.add_argument('model_id', help='model id')
# parser.add_argument('file_path', help='file path to image')
# parser.add_argument('score_threshold', help='threshold')
# args = parser.parse_args()
# print(args)

# from google.cloud import automl_v1beta1 as automl
#
# project_id = args.project_id
# compute_region = 'west-central1'
# model_id = args.model_id
# file_path = args.file_path
# score_threshold = args.score_threshold
#
# automl_client = automl.AutoMlClient()
#
# # Get the full path of the model.
# model_full_id = automl_client.model_path(
#     project_id, compute_region, model_id
# )
#
# # Create client for prediction service.
# prediction_client = automl.PredictionServiceClient()
#
# # Read the image and assign to payload.
# with open(file_path, "rb") as image_file:
#     content = image_file.read()
# payload = {"image": {"image_bytes": content}}
#
# # params is additional domain-specific parameters.
# # score_threshold is used to filter the result
# # Initialize params
# params = {}
# if score_threshold:
#     params = {"score_threshold": score_threshold}
#
# response = prediction_client.predict(model_full_id, payload, params)
# print("Prediction results:")
# for result in response.payload:
#     print("Predicted class name: {}".format(result.display_name))
#     print("Predicted class score: {}".format(result.classification.score))

# 'content' is base-64-encoded image data.

import sys

from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

def get_prediction(content, project_id, model_id, score_threshold):
  prediction_client = automl_v1beta1.PredictionServiceClient()

  name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
  payload = {'image': {'image_bytes': content }}
  params = {"score_threshold": score_threshold}
  request = prediction_client.predict(name, payload, params)
  return request  # waits till request is returned

if __name__ == '__main__':
  file_path = sys.argv[1]
  score_threshold = 0.5
  if sys.argv[2]:
      score_threshold = sys.argv[2]
  print('file: {}, score: {}'.format(file_path, score_threshold))
  project_id = '361695417732'
  # model_id = 'ICN2971940347459928064' # multi class
  # model_id = 'ICN7941381061286559744' # multi label
  model_id = 'ICN8322357442264432640'

  with open(file_path, 'rb') as ff:
    content = ff.read()

  print(get_prediction(content, project_id, model_id, score_threshold))
