# Example data (replace with your actual data)
item_prices = [2.5, 1.0, 3.0]  # Prices of each item
quantities = [3, 2, 1]  # Quantities of each item
discount_rate = 10  # Discount rate in percentage
tax_rate = 5  # Tax rate in percentage

# Calculate total cost before discount
subtotal = sum(item_price * quantity for item_price, quantity in zip(item_prices, quantities))

# Calculate discount amount
discount_amount = (discount_rate / 100) * subtotal

# Calculate total cost after discount
total_after_discount = subtotal - discount_amount

# Calculate tax amount
tax_amount = (tax_rate / 100) * total_after_discount

# Calculate final total cost
final_total = total_after_discount + tax_amount

# Print the result
print("Subtotal: $", subtotal)
print("Discount: $", discount_amount)
print("Total after discount: $", total_after_discount)
print("Tax: $", tax_amount)
print("Final Total: $", final_total)
