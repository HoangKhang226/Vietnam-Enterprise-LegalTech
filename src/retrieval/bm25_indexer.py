"""Vietnamese Legal Tokenizer for BM25"""
import re
from typing import List

LEGAL_PHRASES = sorted([
    "doanh nghiệp nhỏ và vừa", "hỗ trợ doanh nghiệp", "hộ kinh doanh",
    "doanh nghiệp siêu nhỏ", "doanh nghiệp vừa", "doanh nghiệp nhỏ",
    "khởi nghiệp sáng tạo", "vốn điều lệ", "đăng ký kinh doanh",
    "đăng ký doanh nghiệp", "giấy chứng nhận đăng ký",
    "người lao động", "hợp đồng lao động", "người sử dụng lao động",
    "thời giờ làm việc", "tiền lương", "an toàn lao động",
    "kỷ luật lao động", "sa thải", "nghỉ phép",
    "bảo hiểm xã hội", "bảo hiểm thất nghiệp", "bảo hiểm y tế",
    "thuế giá trị gia tăng", "thuế thu nhập doanh nghiệp",
    "thuế thu nhập cá nhân", "quản lý thuế", "khai thuế", "nộp thuế",
    "chậm nộp thuế", "hoàn thuế", "miễn thuế", "giảm thuế",
    "hóa đơn điện tử", "hóa đơn chứng từ",
    "báo cáo tài chính", "kế toán", "kiểm toán",
    "sở hữu trí tuệ", "nhãn hiệu", "bản quyền", "sáng chế",
    "kiểu dáng công nghiệp", "chỉ dẫn địa lý",
    "hợp đồng", "hợp đồng mua bán", "hợp đồng dịch vụ",
    "thương mại điện tử", "giao dịch điện tử",
    "xử phạt vi phạm hành chính", "khắc phục hậu quả",
    "vi phạm hành chính", "biện pháp khắc phục",
    "mặt bằng sản xuất", "đất đai", "thuê đất", "quyền sử dụng đất",
    "bảo lãnh tín dụng", "quỹ bảo lãnh", "tín dụng",
    "đấu thầu", "nhà thầu",
    "nghị định", "thông tư", "quyết định", "nghị quyết",
    "văn bản pháp luật", "quy định pháp luật",
    "cấp phép", "thu hồi", "đình chỉ", "tạm đình chỉ",
    "giấy phép", "chứng chỉ hành nghề",
], key=len, reverse=True)

def legal_tokenize(text: str) -> List[str]:
    lowered = text.lower()
    legal_ids = re.findall(r"\b\d{1,3}/\d{4}/[\wđ-]+\b", lowered)
    phrases = [p.replace(" ", "_") for p in LEGAL_PHRASES if p in lowered]
    words = [w for w in re.findall(r"[0-9a-zà-ỹđ]+", lowered) if len(w) > 1 or w.isdigit()]
    return legal_ids + words + phrases
