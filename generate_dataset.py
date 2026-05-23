import pandas as pd

def generate():
    print("Generating synthetic_drug_conversations.csv from original files...")
    
    # Load emoji dataset
    try:
        emoji_df = pd.read_csv('synthetic_emoji_dataset.csv')
        emoji_df['message_text'] = emoji_df['message']
        emoji_df['risk_level'] = emoji_df['suspicious'].apply(lambda x: 'high' if x == 1 else 'low')
        emoji_df['message_type'] = 'general'
        emoji_df = emoji_df[['platform', 'message_type', 'message_text', 'risk_level']]
    except FileNotFoundError:
        print("synthetic_emoji_dataset.csv not found.")
        emoji_df = pd.DataFrame()
        
    # Load slang dataset
    try:
        slang_df = pd.read_csv('synthetic_slang_dataset.csv')
        if 'message' in slang_df.columns:
            slang_df['message_text'] = slang_df['message']
        else:
            slang_df['message_text'] = slang_df.get('text', '')
            
        slang_df['risk_level'] = slang_df['suspicious'].apply(lambda x: 'high' if x == 1 else 'low')
        slang_df['message_type'] = 'general'
        if 'platform' not in slang_df.columns:
            slang_df['platform'] = 'Telegram'
        slang_df = slang_df[['platform', 'message_type', 'message_text', 'risk_level']]
    except FileNotFoundError:
        print("synthetic_slang_dataset.csv not found.")
        slang_df = pd.DataFrame()

    # Combine and save
    combined = pd.concat([emoji_df, slang_df], ignore_index=True)
    if not combined.empty:
        # Generate some 'medium' and 'low' normal messages to ensure balance if they don't exist
        normal_data = pd.DataFrame({
            'platform': ['Telegram', 'Discord', 'WhatsApp', 'Reddit'] * 250,
            'message_type': ['general'] * 1000,
            'message_text': ['Hello how are you', 'What time is the meeting', 'See you tomorrow', 'Can you call me later'] * 250,
            'risk_level': ['low'] * 1000
        })
        combined = pd.concat([combined, normal_data], ignore_index=True)
        
        combined.to_csv('synthetic_drug_conversations.csv', index=False)
        print(f"Generated synthetic_drug_conversations.csv with {len(combined)} rows.")
    else:
        print("No source files found, cannot generate.")

if __name__ == "__main__":
    generate()
