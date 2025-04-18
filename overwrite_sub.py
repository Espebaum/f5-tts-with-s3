import boto3
import pandas as pd
import io

def update_multiple_csvs_with_sub(bucket, base_prefix, start=2, end=26, region="ap-northeast-2"):
    s3 = boto3.client("s3", region_name=region)

    for i in range(start, end + 1):
        sub_value = f"KO-B{str(i).zfill(6)}"
        key = f"{base_prefix}/{sub_value}.csv"
        s3_path = f"s3://{bucket}/{key}"
        print(f"🔄 Updating: {s3_path} with sub = {sub_value}")

        try:
            # 1. S3에서 파일 읽기
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read()

            # 2. DataFrame 로딩 및 sub 열 추가
            df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
            df["sub"] = sub_value

            # 3. 다시 S3에 덮어쓰기
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False, encoding="utf-8-sig")
            buffer.seek(0)
            s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())

            print(f"✅ Successfully updated {s3_path}")
        except Exception as e:
            print(f"❌ Failed to process {s3_path}: {e}")

# 실행
update_multiple_csvs_with_sub(
    bucket="deeploading-aidubbing",
    base_prefix="Emilia-Dataset/Emilia-YODAS/KO/metadata",
    start=0,
    end=207
)
