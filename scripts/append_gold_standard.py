from src.config.logger import logger
import json

new_gold_standard = [
  {
    "query": "Công ty cần nộp lệ phí duy trì hiệu lực Bằng bảo hộ giống cây trồng vào thời điểm nào?",
    "expected_doc_ids": ["50/2005/QH11", "31/2023/NĐ-CP"]
  },
  {
    "query": "Khi yêu cầu cơ quan hải quan kiểm soát hàng hóa xâm phạm quyền sở hữu trí tuệ, công ty có nghĩa vụ gì?",
    "expected_doc_ids": ["50/2005/QH11", "13/2015/TT-BTC"]
  },
  {
    "query": "Khi công ty sử dụng bản ghi âm, ghi hình đã công bố cho hoạt động kinh doanh thì phải thực hiện nghĩa vụ gì đối với chủ sở hữu quyền tác giả?",
    "expected_doc_ids": ["50/2005/QH11", "131/2013/NĐ-CP"]
  },
  {
    "query": "Khi nộp đơn đăng ký chỉ dẫn địa lý, công ty cần chuẩn bị những tài liệu và thông tin gì?",
    "expected_doc_ids": ["50/2005/QH11", "11/2015/TT-BKHCN"]
  },
  {
    "query": "Khi công ty bị kiện vì xâm phạm quyền sở hữu trí tuệ thì Tòa án có thể áp dụng những biện pháp dân sự nào?",
    "expected_doc_ids": ["50/2005/QH11", "13/2015/TT-BTC"]
  },
  {
    "query": "Điều kiện để tên thương mại của công ty được bảo hộ là gì?",
    "expected_doc_ids": ["50/2005/QH11", "11/2015/TT-BKHCN"]
  },
  {
    "query": "Để được coi là có khả năng phân biệt, tên thương mại của công ty cần đáp ứng những điều kiện nào?",
    "expected_doc_ids": ["50/2005/QH11", "11/2015/TT-BKHCN"]
  },
  {
    "query": "Công ty có bị coi là vi phạm pháp luật nếu cho người khác sử dụng mã số thuế của mình không?",
    "expected_doc_ids": ["38/2019/QH14", "125/2020/NĐ-CP"]
  },
  {
    "query": "Hồ sơ đăng ký quyền tác giả cho tác phẩm của công ty bao gồm những giấy tờ gì?",
    "expected_doc_ids": ["50/2005/QH11", "21/2018/TT-BYT"]
  },
  {
    "query": "Những dấu hiệu nào không được bảo hộ với danh nghĩa nhãn hiệu khi công ty đăng ký thương hiệu?",
    "expected_doc_ids": ["50/2005/QH11", "11/2015/TT-BKHCN"]
  },
  {
    "query": "Công ty có quyền ngăn cấm người khác sử dụng đối tượng sở hữu công nghiệp của mình trong trường hợp nào?",
    "expected_doc_ids": ["50/2005/QH11"]
  },
  {
    "query": "Công ty có bị phạt nếu trả lương thử việc cho nhân viên thấp hơn 85% mức lương chính thức không?",
    "expected_doc_ids": ["45/2019/QH14", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty không lập hồ sơ vệ sinh môi trường lao động đối với các yếu tố có hại thì sẽ bị phạt bao nhiêu tiền?",
    "expected_doc_ids": ["84/2015/QH13", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty dùng hình thức phạt tiền hoặc cắt lương thay cho kỷ luật lao động thì sẽ bị xử lý như thế nào?",
    "expected_doc_ids": ["45/2019/QH14", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty không khai báo với Sở Lao động - Thương binh và Xã hội khi đưa máy móc có yêu cầu nghiêm ngặt về an toàn lao động vào sử dụng thì bị phạt bao nhiêu?",
    "expected_doc_ids": ["84/2015/QH13", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty không tiến hành quan trắc môi trường lao động để kiểm soát tác hại sức khỏe cho nhân viên thì sẽ bị phạt bao nhiêu tiền?",
    "expected_doc_ids": ["84/2015/QH13", "12/2022/NĐ-CP"]
  },
  {
    "query": "Công ty có bị phạt tiền nếu bố trí nhân viên làm công việc có yêu cầu nghiêm ngặt về an toàn mà không có thẻ an toàn không?",
    "expected_doc_ids": ["84/2015/QH13", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty không báo cáo hoặc báo cáo không đúng thời hạn về công tác an toàn, vệ sinh lao động thì bị phạt bao nhiêu tiền?",
    "expected_doc_ids": ["84/2015/QH13", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty không lập sổ theo dõi riêng khi sử dụng lao động chưa thành niên thì sẽ bị xử phạt như thế nào?",
    "expected_doc_ids": ["45/2019/QH14", "12/2022/NĐ-CP"]
  },
  {
    "query": "Nếu công ty giữ bản chính văn bằng, chứng chỉ của nhân viên thì biện pháp khắc phục hậu quả là gì?",
    "expected_doc_ids": ["45/2019/QH14", "12/2022/NĐ-CP"]
  },
  {
    "query": "Công ty có bị phạt nếu không kịp thời sơ cứu hoặc cấp cứu cho nhân viên bị tai nạn lao động không?",
    "expected_doc_ids": ["84/2015/QH13", "12/2022/NĐ-CP"]
  },
  {
    "query": "Khi lập hóa đơn cho khách hàng nước ngoài đến Việt Nam, công ty có thể dùng thông tin gì thay thế cho địa chỉ người mua?",
    "expected_doc_ids": ["78/2021/TT-BTC", "123/2020/NĐ-CP"]
  },
  {
    "query": "Công ty có bị phạt nếu yêu cầu người lao động khuyết tật nặng làm thêm giờ mà họ không đồng ý không?",
    "expected_doc_ids": ["45/2019/QH14", "144/2013/NĐ-CP", "12/2022/NĐ-CP"]
  },
  {
    "query": "Định dạng của hóa đơn điện tử hiện nay được quy định sử dụng ngôn ngữ nào?",
    "expected_doc_ids": ["78/2021/TT-BTC"]
  },
  {
    "query": "Công ty có bị phạt nếu không cho cán bộ công đoàn cấp trên vào doanh nghiệp để tuyên truyền, hướng dẫn người lao động thành lập công đoàn không?",
    "expected_doc_ids": ["82/2020/NĐ-CP", "12/2022/NĐ-CP"]
  },
  {
    "query": "Những doanh nghiệp nhỏ và vừa nào được miễn trả tiền dịch vụ hóa đơn điện tử có mã của cơ quan thuế?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Công ty muốn lập hóa đơn điện tử có mã của cơ quan thuế thông qua tổ chức cung cấp dịch vụ thì thực hiện như thế nào?",
    "expected_doc_ids": ["78/2021/TT-BTC"]
  },
  {
    "query": "Khi phát hiện hóa đơn điện tử đã gửi cho khách hàng bị sai tên hoặc địa chỉ người mua thì công ty phải xử lý như thế nào?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Công ty muốn sử dụng hóa đơn điện tử không có mã của cơ quan thuế thì cần điều kiện gì?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Trong trường hợp nào doanh nghiệp bị cơ quan thuế cưỡng chế ngừng sử dụng hóa đơn mà vẫn được cấp hóa đơn điện tử có mã theo từng lần phát sinh?",
    "expected_doc_ids": ["126/2020/NĐ-CP", "123/2020/NĐ-CP"]
  },
  {
    "query": "Khi sử dụng hóa đơn điện tử có mã của cơ quan thuế, công ty có trách nhiệm gì trong việc quản lý tài khoản?",
    "expected_doc_ids": ["78/2021/TT-BTC"]
  },
  {
    "query": "Khi công ty gặp sự cố không sử dụng được hóa đơn điện tử có mã của cơ quan thuế thì phải làm gì?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Những trường hợp nào công ty được phép mua hóa đơn do cơ quan thuế đặt in?",
    "expected_doc_ids": ["151/2017/NĐ-CP", "123/2020/NĐ-CP"]
  },
  {
    "query": "Khi sử dụng hóa đơn điện tử có mã của cơ quan thuế, công ty có trách nhiệm gì đối với người mua?",
    "expected_doc_ids": ["78/2021/TT-BTC"]
  },
  {
    "query": "Hạn cuối để công ty nộp báo cáo tình hình sử dụng hóa đơn đặt in mua của cơ quan thuế theo quý là khi nào?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Khi chuyển sang sử dụng hóa đơn điện tử thì công ty phải xử lý các hóa đơn giấy đã mua của cơ quan thuế như thế nào?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Trong những trường hợp nào thì công ty bị buộc phải ngừng sử dụng hóa đơn điện tử?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Để đăng ký sử dụng biên lai điện tử thì tổ chức thu phí, lệ phí phải thực hiện qua kênh nào?",
    "expected_doc_ids": ["303/2016/TT-BTC", "78/2021/TT-BTC"]
  },
  {
    "query": "Khi công ty thông báo với cơ quan thuế về việc không tiếp tục sử dụng hóa đơn thì thời hạn chậm nhất để tiêu hủy là bao lâu?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Công ty xử lý thế nào khi hóa đơn mua của cơ quan thuế đã giao cho khách hàng nhưng bị sai tên và địa chỉ người mua?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "39/2014/TT-BTC"]
  },
  {
    "query": "Khi tạm ngừng cung cấp thông tin hóa đơn điện tử, Tổng cục Thuế phải thông báo những nội dung gì cho doanh nghiệp?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Thời hạn sử dụng tài khoản truy cập Cổng thông tin điện tử để sử dụng thông tin hóa đơn điện tử là bao lâu?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Khi bán hàng hóa và cung cấp dịch vụ cho khách hàng, công ty có nghĩa vụ gì về việc lập và giao hóa đơn?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Khi sử dụng thông tin hóa đơn điện tử, công ty có trách nhiệm gì trong việc bảo mật tài khoản truy cập do Tổng cục Thuế cấp?",
    "expected_doc_ids": ["78/2021/TT-BTC"]
  },
  {
    "query": "Công ty cần bảo quản và lưu trữ hóa đơn điện tử bằng phương thức nào?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Khi mua hàng hóa và dịch vụ, công ty có trách nhiệm gì đối với việc lập hóa đơn của người bán?",
    "expected_doc_ids": ["123/2020/NĐ-CP"]
  },
  {
    "query": "Trong những trường hợp nào tài khoản truy cập Cổng thông tin điện tử của công ty sẽ bị Tổng cục Thuế thu hồi?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Khi công ty cung cấp dịch vụ mà khách hàng trả tiền trước thì thời điểm lập hóa đơn là khi nào?",
    "expected_doc_ids": ["123/2020/NĐ-CP", "78/2021/TT-BTC"]
  },
  {
    "query": "Nếu công ty cung cấp hồ sơ pháp lý đăng ký thuế cho cơ quan thuế chậm từ 5 ngày làm việc trở lên thì bị phạt bao nhiêu?",
    "expected_doc_ids": ["125/2020/NĐ-CP"]
  },
  {
    "query": "Nếu công ty không lập hóa đơn khi bán hàng hóa, dịch vụ thì sẽ bị xử phạt và phải khắc phục hậu quả như thế nào?",
    "expected_doc_ids": ["125/2020/NĐ-CP"]
  }
]

import os

dataset_path = 'd:/Project/Legal AI/data/eval_dataset.json'

with open(dataset_path, 'r', encoding='utf-8') as f:
    existing_data = json.load(f)

# Combine the unique objects
existing_queries = {item['query'] for item in existing_data}
added_count = 0

for item in new_gold_standard:
    if item['query'] not in existing_queries:
        existing_data.append(item)
        added_count += 1

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(existing_data, f, ensure_ascii=False, indent=2)

logger.info(f"Đã thêm thành công {added_count} câu hỏi vào Gold Standard Dataset. Tổng số: {len(existing_data)} câu.")
