class Decode_util():
    @staticmethod
    def b8(bitstream):
        value = bitstream[:8]
        bitstream = bitstream[8:]
        return value, bitstream
        
    @staticmethod
    def uv(bitstream, num_bits):
        value = 0
        tmp_bitstream = list(bitstream[:num_bits])
        bitstream = bitstream[num_bits:]
        for _ in range(num_bits):
            value = (value << 1) + int(tmp_bitstream.pop(0))
        return value, bitstream
    
    @staticmethod
    def fn(bitstream, num_bits):
    # Generate a fixed-pattern bit string of length n
        value = bitstream[:num_bits]
        bitstream = bitstream[num_bits:]
        return int(value, 2), bitstream
    
    @staticmethod
    def uev(bitstream):
        zeros = 0
        while True:
            tmp_value = bitstream[0]
            bitstream = bitstream[1:]
            if tmp_value == '1':  # Assuming bitstream is a list of '0's and '1's
                break
            zeros += 1
        value = 1 << zeros  # 2^zeros
        for i in range(zeros):
            tmp_value = bitstream[0]
            bitstream = bitstream[1:]
            value += int(tmp_value) << (zeros - 1 - i)
        return value - 1, bitstream

    @staticmethod
    def sev(bitstream):
        # Decode using the previously defined uev function
        ue_value, bitstream = Decode_util.uev(bitstream)  # Assume uev is already defined as shown earlier
        # Map to signed value
        if ue_value % 2 == 0:
            return (-ue_value // 2), bitstream
        else:
            return ((ue_value + 1) // 2), bitstream
   