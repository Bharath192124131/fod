average_price_per_location = property_data.groupby('location')['listing_price'].mean()
print("Average Listing Price per Location:")
print(average_price_per_location)
properties_more_than_four_bedrooms = property_data[property_data['bedrooms'] > 4]
num_properties_more_than_four_bedrooms = len(properties_more_than_four_bedrooms)
print("Number of Properties with More Than Four Bedrooms:", num_properties_more_than_four_bedrooms)
property_with_largest_area = property_data.loc[property_data['area_sqft'].idxmax()]
print("Property with the Largest Area:")
print(property_with_largest_area)
