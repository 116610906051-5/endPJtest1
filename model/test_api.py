from main import *

# Test 1: ข่าวปลอม
test1 = predict(News(text='กรมพัฒนาธุรกิจการค้าอนุญาตใบทะเบียนพาณิชย์รายบุคคล ประกอบธุรกิจเงินกู้นอกระบบแบบออนไลน์'))
print("Test 1 (ควรเป็นข่าวปลอม):")
print(f"  Label: {test1['label']}")
print(f"  Confidence: {test1['confidence']}%")
print(f"  Raw Score: {test1['raw_score']}")

print()

# Test 2: ข่าวจริง
test2 = predict(News(text='นายกรัฐมนตรีเปิดเผยนโยบายเศรษฐกิจใหม่ในการประชุมคณะรัฐมนตรี'))
print("Test 2 (ควรเป็นข่าวจริง):")
print(f"  Label: {test2['label']}")
print(f"  Confidence: {test2['confidence']}%")
print(f"  Raw Score: {test2['raw_score']}")
