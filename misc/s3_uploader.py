import os
import boto3
import botocore

aws_access_key_id = "my-key-id"
aws_secret_access_key = "my-secret-key"
endpoint_url = "https://my-endpoint-url"
region_name = "us-east-1"
bucket_name = "my-bucket"

if not all([aws_access_key_id, aws_secret_access_key, endpoint_url, region_name, bucket_name]):
    raise ValueError("One or data connection variables are empty.  "
                     "Please check your data connection to an S3 bucket.")

session = boto3.session.Session(aws_access_key_id=aws_access_key_id,
                                aws_secret_access_key=aws_secret_access_key)

s3_resource = session.resource(
    's3',
    config=botocore.client.Config(signature_version='s3v4'),
    endpoint_url=endpoint_url,
    region_name=region_name)

bucket = s3_resource.Bucket(bucket_name)


def upload_directory_to_s3(local_directory, s3_prefix):
    num_files = 0
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, local_directory)
            if ".git" in relative_path:
                print(f"skipping {relative_path}")
                continue
            s3_key = os.path.join(s3_prefix, relative_path)
            print(f"{file_path} -> {s3_key}")
            bucket.upload_file(file_path, s3_key)
            num_files += 1
    return num_files


def list_objects(prefix):
    filter = bucket.objects.filter(Prefix=prefix)
    for obj in filter.all():
        print(obj.key)
        

def count_objects(prefix):
    filter = bucket.objects.filter(Prefix=prefix)
    count = 0
    for obj in filter.all():
        count += 1
    print(f"{count} objects found in {prefix}")
    return count


def delete_objects(prefix):
    filter = bucket.objects.filter(Prefix=prefix)
    for obj in filter.all():
        print(f"deleting {obj.key}")    
        obj.delete()


def download_objects(prefix, local_dest_dir):
    for obj in bucket.objects.filter(Prefix=prefix).all():
        stripped_key = obj.key.replace(prefix, '')
        if stripped_key.startswith('/'):
            stripped_key = stripped_key[1:]
        new_file_path = os.path.join(local_dest_dir, stripped_key)

        try:
            os.makedirs(os.path.dirname(new_file_path))
        except OSError:
            pass

        if new_file_path.endswith('/'):
            continue

        if os.path.exists(new_file_path):
            os.remove(new_file_path)

        print(f"downloading {obj.key} -> {new_file_path}")

        bucket.download_file(obj.key, new_file_path)

upload_directory_to_s3("<path-to-best-performed-checkpoint>", "my-final-model")
