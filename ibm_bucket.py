import ibm_boto3
from ibm_botocore.client import Config
import cv2

# Constants for IBM COS values
COS_ENDPOINT = "https://s3.au-syd.cloud-object-storage.appdomain.cloud" # Current list avaiable ath ttps://control.cloud-object-storage.cloud.ibm.com/v2/endpoints
COS_API_KEY_ID = "3h2SAwb7olaQnpHVC1NJeHwFT-xTodI9gtCJQ9HIkaIp"
COS_AUTH_ENDPOINT = "https://iam.bluemix.net/oidc/token"
COS_RESOURCE_CRN = "crn:v1:bluemix:public:cloud-object-storage:global:a/c2bd62b041f74646a81c5445fbf94f81:3e71e118-30e1-4e9e-b795-165f3959b051:bucket:extract-sig"
COS_BUCKET_LOCATION = "au-syd"

# Create resource
cos = ibm_boto3.client("s3",
    ibm_api_key_id=COS_API_KEY_ID,
    ibm_service_instance_id=COS_RESOURCE_CRN,
    ibm_auth_endpoint=COS_AUTH_ENDPOINT,
    config=Config(signature_version="oauth"),
    endpoint_url=COS_ENDPOINT
)


cos.upload_file(Filename='Signature_Extraction/9.jpg',Bucket='extract-sig',Key='9.jpg')

cos.download_file(Bucket="extract-sig",Key='9.jpg',Filename='9.jpg')

