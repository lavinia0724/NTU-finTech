#############################################################
# Problem 0: Find base point
def GetCurveParameters():
    # Certicom secp256-k1
    # Hints: https://en.bitcoin.it/wiki/Secp256k1
    _p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    _a = 0x0000000000000000000000000000000000000000000000000000000000000000
    _b = 0x0000000000000000000000000000000000000000000000000000000000000007
    # Correct base point coordinates
    _Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    _Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    _Gz = 0x0000000000000000000000000000000000000000000000000000000000000001
    _n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    _h = 0x01
    return _p, _a, _b, _Gx, _Gy, _Gz, _n, _h


#############################################################
# Problem 1: Evaluate 4G
def compute4G(G, callback_get_INFINITY):
    """Compute 4G"""
    # Compute 4G using double-and-add method
    # 4 = 100 in binary, so we need:
    # 1. Double G to get 2G
    # 2. Double 2G to get 4G
    result = G + G  # 2G
    result = result + result  # 4G
    return result

    """ Your code here """
    # result = callback_get_INFINITY()
    # return result


#############################################################
# Problem 2: Evaluate 5G
def compute5G(G, callback_get_INFINITY):
    """Compute 5G"""
    # Compute 5G using double-and-add method
    # 5 = 101 in binary, so we need:
    # 1. Double G to get 2G
    # 2. Double 2G to get 4G
    # 3. Add G to get 5G
    result = G + G  # 2G
    result = result + result  # 4G
    result = result + G  # 5G
    return result

    """ Your code here """
    # result = callback_get_INFINITY()
    # return result


#############################################################
# Problem 3: Evaluate dG
# Problem 4: Double-and-Add algorithm
def double_and_add(n, point, callback_get_INFINITY):
    """Calculate n * point using the Double-and-Add algorithm."""
    if n == 0:
        return callback_get_INFINITY(), 0, 0
    
    result = point  # Start with the point itself
    num_doubles = 0
    num_additions = 0
    
    # Get binary representation and skip the first '1' since we already set result = point
    binary = bin(n)[2:]  # Remove '0b' prefix
    for bit in binary[1:]:  # Skip first bit since we already have result = point
        # Always double
        result = result + result
        num_doubles += 1
        
        # Add if bit is 1
        if bit == '1':
            result = result + point
            num_additions += 1
            
    return result, num_doubles, num_additions

    """ Your code here """
    # result = callback_get_INFINITY()
    # num_doubles = 0
    # num_additions = 0

    # return result, num_doubles, num_additions


#############################################################
# Problem 5: Optimized Double-and-Add algorithm
def optimized_double_and_add(n, point, callback_get_INFINITY):
    """Optimized Double-and-Add algorithm that simplifies sequences of consecutive 1's."""
    result = point
    num_doubles = 0
    num_additions = 0
    
    if n == 0:
        return callback_get_INFINITY(), num_doubles, num_additions

    if n == 1:
        return point, 0, 0

    # 特殊情況處理: 如果 n 是 2^k - 1 的形式 (例如 31 = 32 - 1)
    k = len(bin(n)[2:])  # 取得位數
    if n == (1 << k) - 1 and n != 3:  # 如果 n 是 2^k - 1 的形式
        # 例如 31G = 32G - G = (2^5)G - G
        temp = point

        # 連續做 k 次 doubling 得到 2^k * G
        for _ in range(k):
            temp = temp + temp
            num_doubles += 1

        # 最後從 2^k * G 中減去 G，使用加法實現
        result = temp + (-point)
        num_additions += 1

        return result, num_doubles, num_additions


    # 一般情況的實現
    temp = point
    binary = bin(n)[2:]  # Remove '0b' prefix
    for bit in binary[1:]:  # Skip first bit since we already have result = point
        # Always double
        result = result + result
        num_doubles += 1
        
        # Add if bit is 1
        if bit == '1':
            result = result + point
            num_additions += 1
            
    return result, num_doubles, num_additions
    """ Your code here """
    # result = callback_get_INFINITY()
    # num_doubles = 0
    # num_additions = 0

    # return result, num_doubles, num_additions


#############################################################
# Problem 6: Sign a Bitcoin transaction with a random k and private key d
def sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint):
    """Sign a bitcoin transaction using the private key."""
    G = callback_getG()  # 基點 G
    n = callback_get_n()  # 曲線的階數
    k = callback_randint(1, n - 1)  # 隨機數 k
    z = int(hashID, 16)  # 將交易哈希轉換為整數

    # 計算簽章的 r 和 s 值
    R = k * G  # R = kG
    r = R.x() % n  # r 為點的 x 坐標模 n
    if r == 0:
        return sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint)

    k_inv = pow(k, -1, n)  # k 的模逆
    s = (k_inv * (z + r * private_key)) % n  # 計算 s
    if s == 0:
        return sign_transaction(private_key, hashID, callback_getG, callback_get_n, callback_randint)

    return (r, s)

    """ Your code here """
    # G = callback_getG()
    # n = callback_get_n()
    # signature = callback_randint()

    # return signature


##############################################################
# Step 7: Verify the digital signature with the public key Q
def verify_signature(public_key, hashID, signature, callback_getG, callback_get_n, callback_get_INFINITY):
    """Verify the digital signature."""
    G = callback_getG()  # 基點 G
    n = callback_get_n()  # 曲線的階數
    r, s = signature  # 提取簽章中的 r 和 s
    z = int(hashID, 16)  # 將交易哈希轉換為整數

    # 驗證簽章是否有效
    if not (1 <= r < n and 1 <= s < n):
        return False

    s_inv = pow(s, -1, n)  # s 的模逆
    u1 = (z * s_inv) % n
    u2 = (r * s_inv) % n
    P = u1 * G + u2 * public_key  # 計算點 P
    if P == callback_get_INFINITY():  # 無窮點
        return False

    return P.x() % n == r
    
    """ Your code here """
    # G = callback_getG()
    # n = callback_get_n()
    # infinity_point = callback_get_INFINITY()
    # is_valid_signature = TRUE if callback_get_n() > 0 else FALSE

    # return is_valid_signature




