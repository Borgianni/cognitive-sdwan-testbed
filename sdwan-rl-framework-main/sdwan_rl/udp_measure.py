"""Bounded UDP sequence/echo worker for the local Mininet clock domain.

Invoked inside hosts by TelemetryCollector. One process handles all streams
concurrently, with common monotonic send/receive deadlines. No RL dependency.
"""
import argparse
import json
import selectors
import socket
import struct
import time

HEADER = struct.Struct("!16sI")


def stamp():
    return time.monotonic()


def packet(token, seq, size):
    return HEADER.pack(bytes.fromhex(token), seq) + bytes(size - HEADER.size)


def decode(data, token, count, size):
    if len(data) != size or len(data) < HEADER.size:
        return None
    tag, seq = HEADER.unpack_from(data)
    return seq if tag == bytes.fromhex(token) and seq < count else None


def run_server(spec):
    end = spec["start_s"] + spec["window_s"] + spec["grace_s"]
    selector = selectors.DefaultSelector()
    result = {}
    seen = {}
    try:
        for name, stream in spec["streams"].items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((spec["server_ip"], stream["port"]))
            sock.setblocking(False)
            selector.register(sock, selectors.EVENT_READ, name)
            seen[name] = set()
            result[name] = {"received_unique": 0, "received_payload_bytes": 0,
                            "echo_sent": 0, "echo_payload_bytes": 0,
                            "first_receive_s": None, "last_receive_s": None}
        print("READY", flush=True)
        while stamp() < end:
            for key, _ in selector.select(max(0.0, end - stamp())):
                sock, name = key.fileobj, key.data
                if stamp() >= end:
                    break
                try:
                    data, addr = sock.recvfrom(65535)
                except BlockingIOError:
                    continue
                stream, row = spec["streams"][name], result[name]
                seq = decode(data, spec["token"], stream["packets"], stream["payload_bytes"])
                if seq is None or seq in seen[name]:
                    continue
                seen[name].add(seq)
                now = stamp()
                row["received_unique"] += 1
                row["received_payload_bytes"] += len(data)
                if row["first_receive_s"] is None:
                    row["first_receive_s"] = now
                row["last_receive_s"] = now
                try:
                    if sock.sendto(data, addr) == len(data):
                        row["echo_sent"] += 1
                        row["echo_payload_bytes"] += len(data)
                except (BlockingIOError, OSError):
                    # Local echo send failure is distinguishable from link loss.
                    pass
        return {"streams": result, "finished_s": stamp()}
    finally:
        for key in list(selector.get_map().values()):
            key.fileobj.close()
        selector.close()


def run_client(spec):
    start = spec["start_s"]
    window = spec["window_s"]
    end = start + window + spec["grace_s"]
    selector = selectors.DefaultSelector()
    result, sent_at, rtts, sockets = {}, {}, {}, {}
    schedule = []
    try:
        for name, stream in spec["streams"].items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((spec["client_ip"], 0))
            sock.setblocking(False)
            selector.register(sock, selectors.EVENT_READ, name)
            sockets[name] = sock
            sent_at[name], rtts[name] = {}, {}
            result[name] = {"sent": 0, "sent_payload_bytes": 0,
                            "first_send_s": None, "last_send_s": None,
                            "first_echo_s": None, "last_echo_s": None,
                            "max_send_lateness_s": 0.0}
            schedule.extend((start + i * window / stream["packets"], name, i)
                            for i in range(stream["packets"]))
        schedule.sort()
        index = 0
        while stamp() < end:
            now = stamp()
            while index < len(schedule) and schedule[index][0] <= now:
                due, name, seq = schedule[index]
                index += 1
                # Do not compensate for missed windows with an out-of-window burst.
                if stamp() >= start + window:
                    continue
                stream, row = spec["streams"][name], result[name]
                data = packet(spec["token"], seq, stream["payload_bytes"])
                before = stamp()
                try:
                    n = sockets[name].sendto(data, (spec["server_ip"], stream["port"]))
                except (BlockingIOError, OSError):
                    continue
                if n != len(data):
                    continue
                sent_at[name][seq] = before
                row["sent"] += 1
                row["sent_payload_bytes"] += n
                if row["first_send_s"] is None:
                    row["first_send_s"] = before
                row["last_send_s"] = before
                row["max_send_lateness_s"] = max(row["max_send_lateness_s"], before - due)
            deadline = min(end, schedule[index][0]) if index < len(schedule) else end
            for key, _ in selector.select(max(0.0, deadline - stamp())):
                sock, name = key.fileobj, key.data
                if stamp() >= end:
                    break
                try:
                    data, addr = sock.recvfrom(65535)
                except BlockingIOError:
                    continue
                stream, row = spec["streams"][name], result[name]
                if addr != (spec["server_ip"], stream["port"]):
                    continue
                seq = decode(data, spec["token"], stream["packets"], stream["payload_bytes"])
                if seq is None or seq not in sent_at[name] or seq in rtts[name]:
                    continue
                now = stamp()
                rtts[name][seq] = (now - sent_at[name][seq]) * 1000
                if row["first_echo_s"] is None:
                    row["first_echo_s"] = now
                row["last_echo_s"] = now
        for name, row in result.items():
            values = list(rtts[name].values())
            row["echo_received_unique"] = len(values)
            row["rtt_ms"] = sum(values) / len(values) if values else None
        return {"streams": result, "finished_s": stamp()}
    finally:
        for sock in sockets.values():
            sock.close()
        selector.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("server", "client"))
    parser.add_argument("spec")
    args = parser.parse_args()
    spec = json.loads(args.spec)
    result = run_server(spec) if args.role == "server" else run_client(spec)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
