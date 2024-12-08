def find_private_key(r, s, k, z, n):
    """Find the private key given r, s, k, z, and n."""
    # s * k - z ≡ r * d (mod n)
    private_key = (s * k - z) * pow(r, -1, n) % n
    return private_key


# Parameters (given or calculated)
r = int("f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9", 16)
s = int("8d89a38eb73d9528e4c1432f88ab9e3a16b4d23f333be3f88a4ce6167c019066", 16)
z = int("4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b", 16)  # Example hashID
k = 123456789  # Assume a known random k
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141  # secp256k1 curve order

# Calculate private key
private_key = find_private_key(r, s, k, z, n)
private_key = int(str(private_key), 16)
print(f"Private Key: {private_key:x}")
