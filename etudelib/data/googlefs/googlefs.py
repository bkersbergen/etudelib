from google.cloud import storage
import os


def upload_to_gcs(local_dir: str, gcs_project_name: str, gcs_bucket_name: str, gcs_dir: str):
    # set up the GCS client
    client = storage.Client(project=gcs_project_name)
    bucket = client.get_bucket(gcs_bucket_name)
    # upload each file in the local directory to GCS
    for root, dirs, files in os.walk(str(local_dir)):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            gcs_path = os.path.join(gcs_dir, relative_path)
            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
