import json
import os
import pandas as pd
from datetime import datetime

OUT_DIR = "./output"

def get_sentiment(rating):
    try:
        r = float(rating or 0)
        if r >= 4.0: return 'positive'
        elif r <= 3.0: return 'negative'
        return 'neutral'
    except:
        return 'neutral'

def extract_attributes(biz):
    attrs = biz.get('attributes', {})
    price = ''
    for key in ['RestaurantsPriceRange2', 'price_range']:
        if key in attrs:
            val = attrs[key]
            try:
                price = '$' * int(val)
                break
            except:
                pass
    attr_count = sum(1 for v in attrs.values() if isinstance(v, bool) and v)
    return {
        'price': price, 
        'attributes_count': attr_count,
        'has_wifi': attrs.get('WiFi', False),
        'has_delivery': attrs.get('RestaurantsDelivery', False),
        'has_takeout': attrs.get('RestaurantsTakeOut', False)
    }

def process_businesses(business_file, sample_size=10000):
    print(f"Loading {os.path.basename(business_file)}...")
    
    if not os.path.exists(business_file):
        print(f"ERROR: {business_file} not found")
        return None
    
    businesses = []
    count = 0
    
    with open(business_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                biz = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            count += 1
            if count % 50000 == 0:
                print(f"  Processed {count:,} lines...")
            
            try:
                row = {
                    'business_id': biz.get('business_id', ''),
                    'business_name': biz.get('name', ''),
                    'city': biz.get('city', ''),
                    'state': biz.get('state', ''),
                    'latitude': biz.get('latitude'),
                    'longitude': biz.get('longitude'),
                    'address': biz.get('address', ''),
                    'postal_code': biz.get('postal_code', ''),
                    'phone': biz.get('phone', ''),
                    'business_rating': biz.get('stars'),
                    'sentiment': get_sentiment(biz.get('stars')),
                    'review_count': biz.get('review_count', 0),
                    'is_open': biz.get('is_open', 0),
                    'categories': biz.get('categories', '')
                }
                
                attrs = extract_attributes(biz)
                row.update(attrs)
                businesses.append(row)
                
                if len(businesses) >= sample_size * 2:
                    print("Hit sample buffer")
                    break
                    
            except Exception:
                continue
    
    if not businesses:
        print("No businesses found")
        return None
    
    try:
        df = pd.DataFrame(businesses)
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sample created: {len(df)} businesses")
        return df
    except Exception as e:
        print(f"DataFrame error: {e}")
        return None

def merge_reviews(df, review_file):
    if not review_file or not os.path.exists(review_file):
        return df
    
    print(f"Processing {os.path.basename(review_file)}...")
    stats = {}
    business_ids = set(df['business_id'])
    count = 0
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                review = json.loads(line.strip())
            except:
                continue
            
            count += 1
            if count % 100000 == 0:
                print(f"  Processed {count:,} reviews...")
            
            bid = review.get('business_id')
            if bid in business_ids:
                if bid not in stats:
                    stats[bid] = {'count': 0, 'stars_sum': 0, 'useful': 0}
                s = stats[bid]
                s['count'] += 1
                s['stars_sum'] += review.get('stars', 0)
                s['useful'] += review.get('useful', 0)
    
    if stats:
        try:
            review_df = pd.DataFrame([{
                'business_id': bid,
                'avg_review_stars': round(v['stars_sum']/max(1, v['count']), 2),
                'review_count_reviews': v['count'],
                'total_useful': v['useful']
            } for bid, v in stats.items()])
            df = df.merge(review_df, on='business_id', how='left')
            print(f"Merged reviews for {len(review_df)} businesses")
        except Exception:
            pass
    
    return df

def main():
    base_path = "data"  # folder with the Yelp JSON files; run from the repo root
    
    files = {
        'business': os.path.join(base_path, 'yelp_academic_dataset_business.json'),
        'review': os.path.join(base_path, 'yelp_academic_dataset_review.json'),
        'checkin': os.path.join(base_path, 'yelp_academic_dataset_checkin.json'),
        'tip': os.path.join(base_path, 'yelp_academic_dataset_tip.json'),
        'user': os.path.join(base_path, 'yelp_academic_dataset_user.json')
    }
    
    print("Yelp Dataset Processor")
    print("Looking for files in:")
    print(files['business'])
    
    # Check business file first
    if not os.path.exists(files['business']):
        print("ERROR: Business file not found!")
        print("Expected:", files['business'])
        return False
    
    print("Files found:")
    for name, path in files.items():
        if os.path.exists(path):
            print(f"  ✓ {name:<8} {os.path.basename(path)}")
        else:
            print(f"  ✗ {name:<8} {os.path.basename(path)}")
    
    # Create output
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
    except:
        pass
    
    print("\nProcessing businesses...")
    df = process_businesses(files['business'], sample_size=10000)
    
    if df is None or df.empty:
        print("Failed to process businesses!")
        return False
    
    print(f"Businesses loaded: {len(df)}")
    
    # Merge available files
    if os.path.exists(files['review']):
        df = merge_reviews(df, files['review'])
    
    print("\nSaving 10K sample...")
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"yelp_10k_sample_{len(df)}_{timestamp}.csv"
        filepath = os.path.join(OUT_DIR, filename)
        df.to_csv(filepath, index=False)
        
        print(f"SUCCESS!")
        print(f"Saved: {filepath}")
        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        
        print("\nSample preview:")
        preview_cols = ['business_name', 'city', 'business_rating', 'review_count', 'sentiment', 'price']
        available_cols = [col for col in preview_cols if col in df.columns]
        print(df[available_cols].head())
        
        return True
        
    except Exception as e:
        print(f"Save error: {e}")
        return False

if __name__ == '__main__':
    try:
        success = main()
        if success:
            print("\nCOMPLETE!")
        else:
            print("\nFAILED!")
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
