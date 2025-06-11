import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from typing import List, Dict, Any

LABEL_COLORS  = ["#FF6347", "#FFD700", "#ADFF2F", "#66CCFF", "#DA70D6"]
CHANNEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c",
                  "#d62728", "#9467bd", "#8c564b"]

def _pretty_xml(root: ET.Element) -> str:
    xml_str = ET.tostring(root, encoding="unicode")
    return "\n".join(
        minidom.parseString(xml_str).toprettyxml(indent="  ").split("\n")[1:]
    )


def _normalize_groups(raw_groups: Any, data_type = 'image') -> List[Dict]:
    """
    将旧有的tuple格式转化为dict格式，并补充缺少的key
    """
    if not raw_groups:
        return []
    label_type = 'TimeSeriesLabels' if data_type.lower() == 'ts' else 'BrushLabels'
    
    if isinstance(raw_groups[0], dict):    
        for group in raw_groups:
            group['smart'] = group.get('smart',False) 
            group['smart'] = group.get('type', label_type) 
        return raw_groups
    
    return [
        {"name": g, "type": label_type, "choices": c, "smart": False}
        for g, c in raw_groups
    ]

def _ensure_auxiliary_labels(groups: list[dict]) -> list[dict]:
    """
    保证using_SAM2需求的三类标签都存在的辅助函数
    """
    # 统计已有类型
    types = {g["type"] for g in groups}
    if "BrushLabels" not in types:        # 理论上不允许，但防御一下
        raise ValueError("SAM 模式至少需要一个 BrushLabels 组")

    # 找到第 1 个 BrushLabels 作为模板
    template = next(g for g in groups if g["type"] == "BrushLabels")

    new_groups = groups[:]

    if "KeyPointLabels" not in types:
        kp = {
            "name": f"{template['name']}_KP",
            "type": "KeyPointLabels",
            "choices": template["choices"],
            "smart": True
        }
        new_groups.append(kp)

    if "RectangleLabels" not in types:
        rect = {
            "name": f"{template['name']}_Rect",
            "type": "RectangleLabels",
            "choices": template["choices"],
            "smart": True
        }
        new_groups.append(rect)

    return new_groups


def _build_label(parent, group: Dict, palette, data_type="image"):
    """
    在 parent 下添加一组标签，
    若data_type=="image",则默认为"BrushLabels";
    若data_type=="ts",则默认为"TimeSeriesLabels"
    """
    if data_type=='ts':
        label_type = 'TimeSeriesLabels'
    if data_type=='image':
        label_type = 'BrushLabels'
    if group.get('type'):
        label_type = group['type']
    node = ET.SubElement(parent, label_type, {
        "name": group["name"], "toName": data_type
    })

    for idx, choice in enumerate(group["choices"]):
        color = palette[idx % len(palette)]
        attrib = {"value": choice, "background": color}
        if data_type.lower() == 'image' and group.get("smart"):
            attrib["smart"] = "true"
        ET.SubElement(node, "Label", attrib=attrib)