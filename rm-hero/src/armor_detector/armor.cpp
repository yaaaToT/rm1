

#include "../../include/armor_detector/armor.h"


Light Armor::extendLight(const Light & light)
{
    Light new_light;
    new_light.top        = light.top;
    new_light.bottom     = light.bottom;
    new_light.length     = light.length;
    new_light.width      = light.width;
    new_light.tilt_angle = light.tilt_angle;

    new_light.angle  = light.angle;
    new_light.center = light.center;

    if (light.size.width < light.size.height)
    {
        new_light.size.height = 2 * light.size.height;
        new_light.size.width  = light.size.width;
    }
    else
    {
        new_light.size.width  = 2 * light.size.width;
        new_light.size.height = light.size.height;
    }
    return new_light;
}


// 排序完成后状态： 左上角为0 顺时针排序
void Armor::sortLightPoints(cv::Point2f points[4],std::vector<cv::Point2f>& new_points)
{
    cv::Point2f temp_point;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = i + 1; j < 4; j++)
        {
            if (points[i].y > points[j].y)
            {
                temp_point = points[i];
                points[i]  = points[j];
                points[j]  = temp_point;
            }
        }
    }
    if (points[0].x > points[1].x)
    {
        temp_point = points[0];
        points[0]  = points[1];
        points[1]  = temp_point;
    }
    if (points[2].x < points[3].x)
    {
        temp_point = points[2];
        points[2]  = points[3];
        points[3]  = temp_point;
    }
    for (size_t i = 0; i < 4; i++)
    {
        new_points.push_back(points[i]);
    }
}

bool ArmorFinder::matchTwoLightBar(const cv::RotatedRect &l, const cv::RotatedRect &r) {
    status = checkAngleDiff(l, r);
    // DLOG_IF(INFO, !status) << "angle differs: left: " << getAngle(l) << " right: " << getAngle(r);
    if(!status) return false;
    status = checkHeightDiff(l, r);
    // DLOG_IF(INFO, !status) << "height differs: " << abs((l.center-r.center).y);
    if(!status) return false;
    // status = checkRatio(l, r);
    // if(!status) return false;
    status = checkHeightMatch(l, r);
    // DLOG_IF(INFO, status) << "height matches: " << l.size.height << " " << r.size.height;
    if(!status) return false;
    return true;
}

bool ArmorFinder::judgeArmor(const ArmorBlob &armor_blob) {
    return armor_blob.rect.size().aspectRatio() <= 5 && armor_blob.rect.size().aspectRatio() > 1.5;
}

bool ArmorFinder::getArmor(const cv::RotatedRect &l, const cv::RotatedRect &r, ArmorBlob& armor) {
    cv::Point2f points_of_rrect[4];
    l.points(points_of_rrect);
    float height = fmax(l.size.width, l.size.height);
    armor.rect = cv::Rect(l.center.x, l.center.y-height/2, r.center.x-l.center.x, height);

    // armor
    // 0 1
    // 3 2


    // 使用的是resnet18，但是1x32x32的输入，将两个点重新选回了灯条的中心，影响不大（之前说是增加BA优化后, 这样可以让yaw轴解算更加准确）
    // TODO 修改代码中的装甲板的四点的选取策略，使用half_length 0.675f进行计算
    // 选取装甲板的中心进行后续求解，这样可以减少装甲板yaw值的误差，同时提供pnp解算的准确度
    if (l.angle > 45)
    {
        armor.corners[0] = (points_of_rrect[0] + points_of_rrect[1]) / 2;
        armor.corners[3] = (points_of_rrect[3] + points_of_rrect[2]) / 2;
    }
    else
    {
        armor.corners[0] = (points_of_rrect[1] + points_of_rrect[2]) / 2;
        armor.corners[3] = (points_of_rrect[0] + points_of_rrect[3]) / 2;
    }
    r.points(points_of_rrect);
    if (r.angle > 45)
    {
        armor.corners[1] = (points_of_rrect[1] + points_of_rrect[0]) / 2;
        armor.corners[2] = (points_of_rrect[2] + points_of_rrect[3]) / 2;
    }
    else
    {
        armor.corners[1] = (points_of_rrect[2] + points_of_rrect[1]) / 2;
        armor.corners[2] = (points_of_rrect[3] + points_of_rrect[0]) / 2;
    }

    if(std::max(armor.corners[0].x, armor.corners[3].x) > std::min(armor.corners[1].x, armor.corners[2].x)) return false;
    return true;
}

// 检查两个灯条角度是否合理;
bool ArmorFinder::checkAngleDiff(const cv::RotatedRect &l, const cv::RotatedRect &r) {
    const float & angle_l = getAngle(l);
    const float & angle_r = getAngle(r);
    // 灯条角度差
    return abs(angle_l-angle_r) < 30;
}

// 获取灯条角度，RotatedRect 角度为 -90到90，角度为负数时表示向右倾斜，为正数时表示向左倾斜
float ArmorFinder::getAngle(const cv::RotatedRect &rrect) {
    // 宽 > 高
    return rrect.size.width > rrect.size.height ? rrect.angle-90 : rrect.angle;
}

// 检查中心距离差 是否大致符合
bool ArmorFinder::checkHeightDiff(const cv::RotatedRect &l, const cv::RotatedRect &r) {
    const cv::Point2f& diff = l.center - r.center;
    return abs(diff.y) < std::min(std::max(l.size.height, l.size.width), std::max(r.size.height, r.size.width))*1.5;
}

bool ArmorFinder::checkHorizontalDistance(const cv::RotatedRect &l, const cv::RotatedRect &r) {
    return false;
}

bool ArmorFinder::checkDislocation(const cv::RotatedRect &l, const cv::RotatedRect &r) {
    return false;
}


// 获取装甲板的极限边界坐标
std::vector<int> ArmorFinder::getExtreme(const ArmorBlob& armor) {

    int x1 = std::min(1280.f, std::max(0.f, std::min(armor.corners[0].x, armor.corners[3].x)));
    int y1 = std::min(1280.f, std::max(0.f, std::min(armor.corners[0].y, armor.corners[1].y) - std::min(armor.rect.height, armor.rect.width)/2.5f));
    int x2 = std::min(1280.f, std::max(0.f, std::max(armor.corners[2].x, armor.corners[1].x)));
    int y2 = std::min(1024.f, std::max(0.f, std::max(armor.corners[2].y, armor.corners[3].y) + std::min(armor.rect.height, armor.rect.width)/2.5f));

    return {x1, y1, x2, y2};
}

// 检查左右灯条的长度不是差别太大
bool ArmorFinder::checkHeightMatch(const cv::RotatedRect &l, const cv::RotatedRect &r) {
    float lh = std::max(l.size.height, l.size.width);
    float rh = std::max(r.size.height, r.size.width);
    return std::min(lh, rh) * 2 > std::max(lh, rh);
}
