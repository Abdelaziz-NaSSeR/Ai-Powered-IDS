from scapy.all import get_if_list

interfaces = get_if_list()
print("Available interfaces:")
for i, iface in enumerate(interfaces):
    print(f"{i+1}. {iface}")

from scapy.arch.windows import get_windows_if_list

for iface in get_windows_if_list():
    print(f"Name: {iface['name']}")
    print(f"Description: {iface['description']}")
    print(f"GUID: {iface['guid']}")
    print("-" * 100)
