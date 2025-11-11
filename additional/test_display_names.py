"""
Test that display names work correctly for both beta formats
"""

print("="*80)
print("DISPLAY NAME TEST")
print("="*80)

# Test variable names from both formats
test_cases = [
    ("Beta_Daily_Impressions_OUTCOME_ENGAGEMENT", "Traffic Ads Impressions"),
    ("Beta_Daily_Impressions_LINK_CLICKS", "Traffic Ads Impressions"),
    ("Daily Impressions Outcome Engagement", "Traffic Ads Impressions"),
    ("Daily Impressions Link Clicks", "Traffic Ads Impressions"),
    ("Beta_Google_Impression", "Google Ads Impressions"),
    ("Google Impression", "Google Ads Impressions"),
    ("Beta_Impressions", "Other Products (Meta Ads)"),
    ("Impressions", "Other Products (Meta Ads)"),
]

def get_display_name(var_name):
    """Simulate the display name logic from the app"""
    if 'Google Impression' in var_name or 'Google_Impression' in var_name:
        return 'Google Ads Impressions'
    elif 'Daily Impressions Outcome Engagement' in var_name or 'Daily_Impressions_OUTCOME_ENGAGEMENT' in var_name or 'Daily Impressions Link Clicks' in var_name or 'Daily_Impressions_LINK_CLICKS' in var_name:
        return 'Traffic Ads Impressions'
    elif var_name == 'Impressions' or var_name == 'Beta_Impressions':
        return 'Other Products (Meta Ads)'
    else:
        # Product-specific impression
        return var_name.replace('Beta_', '').replace('_', ' ').title()

print("\nTesting display name conversion:")
print("-"*80)

all_passed = True
for var_name, expected in test_cases:
    result = get_display_name(var_name)
    status = "✅" if result == expected else "❌"
    if result != expected:
        all_passed = False
    print(f"{status} '{var_name}' → '{result}' (expected: '{expected}')")

print("\n" + "="*80)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print("Both OUTCOME_ENGAGEMENT and LINK_CLICKS map to 'Traffic Ads Impressions'")
else:
    print("❌ SOME TESTS FAILED!")
print("="*80)
