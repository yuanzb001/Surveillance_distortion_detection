import socket
import sys
import struct

def parse_rtp_packet(packet):
    # RTP 数据包的前12个字节是固定头部
    header = struct.unpack_from('!BBHII', packet, 0)
    
    version = header[0] >> 6
    padding = (header[0] >> 5) & 0x01
    extension = (header[0] >> 4) & 0x01
    csrc_count = header[0] & 0x0F
    marker = (header[1] >> 7) & 0x01
    payload_type = header[1] & 0x7F
    sequence_number = header[2]
    timestamp = header[3]
    ssrc = header[4]

    # print(f"Version: {version}, Padding: {padding}, Extension: {extension}, CSRC Count: {csrc_count}")
    # print(f"Marker: {marker}, Payload Type: {payload_type}, Sequence Number: {sequence_number}, Timestamp: {timestamp}, SSRC: {ssrc}")

    header_length = 12 + (csrc_count * 4)
    
    # 如果有扩展头部，进一步解析（此处未展示）
    
    # RTP 负载数据位于 RTP 头部之后
    payload_data = packet[header_length:]

    return payload_data

# def reassemble_nalu(rtp_packets):
    # nalu = bytearray()
    # start_found = False

    # for packet in rtp_packets:
    #     # 假设packet是RTP负载数据
    #     rtp_payload = packet[12:]  # 跳过RTP头

    #     # 检查是否为分片的NALU起始
    #     if rtp_payload[0] & 0x1F == 28:  # FU-A类型
    #         fu_header = rtp_payload[1]
    #         start_bit = fu_header >> 7
    #         nal_header = (rtp_payload[0] & 0xE0) | (fu_header & 0x1F)

    #         # 起始片段
    #         if start_bit:
    #             nalu = bytearray([nal_header])  # 使用NALU头重建NALU
    #             nalu.extend(rtp_payload[2:])  # 添加负载数据
    #             start_found = True
    #         # 中间或结束片段
    #         elif start_found:
    #             nalu.extend(rtp_payload[2:])
    #     else:
    #         # 直接封装的NALU
    #         nalu = bytearray(rtp_payload)

    # return nalu

monitor_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

RTP_PORT = 23450

try:
    monitor_socket.bind(('', RTP_PORT))
except socket.error as msg:
    print(f'Bind failed. Error Code : {str(msg[0])} Message {msg[1]}')
    sys.exit()

print(f'Socket bind complete. Listening on port {RTP_PORT}')

nal_list = []
nalu = bytearray()

# start monitor RTP
packet_count = 0
while True:
    packet_count += 1
# for i in range(100):
    # 接收数据
    data, addr = monitor_socket.recvfrom(4096)  # 缓冲区大小设置为4096字节
    raw_data = parse_rtp_packet(data)
    print(f'Received packet from {addr[0]}:{addr[1]}; length of playload data is {len(raw_data)}')
    # print(type(raw_data))
    slice_flag = (raw_data[0] & 0x1F)
    if slice_flag == 28:
        # print('FU-A type')
        fu_header = raw_data[1]
        start_bit = fu_header >> 7
        end_bit = (fu_header >> 6) & 0x01
        nal_header = (raw_data[0] & 0xE0) | (fu_header & 0x1F)
        # 起始片段
        if start_bit:
            print('nal Unit Start !!!!!!!!!!!!!!!!')
            nalu = bytearray([nal_header])  # 使用NALU头重建NALU
            nalu.extend(raw_data[2:])  # 添加负载数据
            start_found = True
        # 中间或结束片段
        else:
            nalu.extend(raw_data[2:])
            if end_bit == 1:               
                nal_list.append(nalu)
                nalu = bytearray()
                print('nal Unit End !!!!!!!!!')
                if packet_count > 500:
                    print('collect finished!!!')
                    break            
    elif slice_flag == 29:
        print('FU-B')
    else:
        print('whole nalu, nalu type is:', slice_flag)
        nal_list.append(bytearray(raw_data))
        if packet_count > 500:
            print('collect finished!!!')
            break  

def write_nalu_to_file(nal_list, output_file):
    with open(output_file, 'wb') as f:
        for nalu in nal_list:
            # 假设这是一个单一NALU模式的RTP负载
            start_code = b'\x00\x00\x00\x01'
            f.write(start_code)
            f.write(nalu)

# 假设rtp_payloads是一个包含RTP负载数据的列表
write_nalu_to_file(nal_list, 'output_test_500RTP_5.h264')