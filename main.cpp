#include <printf.h>
#include "fast-cpp-csv-parser/csv.h"
#include <string>
#include <algorithm>
#include <cmath>

#include <QVector>
#include <QDebug>

using namespace std;

struct Rect
{
    double x0;
    double y0;
    double x1;
    double y1;
    double confidence;
    bool dropped {false};
};

struct ImageRects {
    std::string image_filename;
    QVector<Rect> rects;
};


double iou(const Rect& box1, const Rect& box2)
{
    double lr = (min(box1.x1, box2.x1) - max(box1.x0, box2.x0)) + 1;
    if (lr > 0){
        double tb = (min(box1.y1, box2.y1) - max(box1.y0, box2.y0)) + 1;

        if (tb > 0)
        {
            double intersection = tb * lr;
            double w1 = box1.x1 - box1.x0 + 1;
            double h1 = box1.y1 - box1.y0 + 1;
            double w2 = box2.x1 - box2.x0 + 1;
            double h2 = box2.y1 - box2.y0 + 1;
            double unionValue = (w1*h1 + w2*h2) - intersection;

            return intersection/unionValue;
        }
    }

    return 0.0;
}


struct ItemsToMerge
{
    int offset;
    double conf;
};


void flexible_nms(int avgItems, QVector<Rect>& items)
{
    double iou_threshold=0.3;
    double iou_merge_threshold=0.75;
    double merge_pow=4.0;

    sort(begin(items), end(items), [](const Rect& r1, const Rect& r2) {return r1.confidence > r2.confidence;});

    int nbItems = items.size();

    QVarLengthArray<ItemsToMerge, 64> itemsToMerge;

    for (int i=0; i<nbItems; i++)
    {
        Rect& row = items[i];
        if (row.dropped)
            continue;

        itemsToMerge.clear();
        itemsToMerge.append(ItemsToMerge{i, pow(row.confidence, merge_pow)});

        for (int j=i+1; j<nbItems; j++)
        {
            if (items[j].dropped)
                continue;

            double cur_iou = iou(row, items[j]);

            if (cur_iou > iou_threshold)
            {
                items[j].dropped = true;
                if (cur_iou > iou_merge_threshold)
                    itemsToMerge.append({j, pow(items[j].confidence, merge_pow) });
            }
        }

        if (itemsToMerge.size() > 1)
        {
            double x0 = 0.0;
            double y0 = 0.0;
            double x1 = 0.0;
            double y1 = 0.0;

            double totalConf = 0.0;

            for (const auto& it: itemsToMerge)
            {
                totalConf += it.conf;

                x0 += items[it.offset].x0 * it.conf;
                x1 += items[it.offset].x1 * it.conf;
                y0 += items[it.offset].y0 * it.conf;
                y1 += items[it.offset].y1 * it.conf;
            }

            row.x0 = x0/totalConf;
            row.x1 = x1/totalConf;
            row.y0 = y0/totalConf;
            row.y1 = y1/totalConf;
        }

        double totalConf = 0.0;
        for (int i=0; i<min(itemsToMerge.size(), avgItems); i++)
        {
            totalConf += items[itemsToMerge[i].offset].confidence;
        }
        row.confidence = totalConf/avgItems;
    }
}



int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        qDebug() << argc;
        qDebug() << "Usage: flexible_nms file_name.csv  > output.csv";
        return 1;
    }

    int avgImages = 3*2*3*3; // 3 sizes * 2 flip * 3 epoch * 3 models

    QMap<std::string, ImageRects> itemsMap;
    int rectsCount = 0;
    int imagesCount = 0;

    int nbFiles = argc-1;
    for (int fnIdx = 0; fnIdx < nbFiles; fnIdx++)
    {
        io::CSVReader<6> in(argv[fnIdx+1]);
        in.read_header(io::ignore_extra_column, "image_filename", "x0", "y0", "x1", "y1", "confidence");

        std::string image_filename;

        qDebug() << "Loading data..." << argv[fnIdx+1];
        Rect r;

        while(in.read_row(image_filename, r.x0, r.y0, r.x1, r.y1, r.confidence)) {
            itemsMap[image_filename].rects.append(r);
            rectsCount++;
        }
    }

    QList<ImageRects> itemsList;

    auto it = itemsMap.constBegin();
    while (it != itemsMap.constEnd()) {
        itemsList.append(it.value());
        itemsList.last().image_filename = it.key();
        ++it;
        imagesCount++;
    }

    qDebug() << "Loaded " << imagesCount << rectsCount;

    int savedRects = 0;
    printf("image_filename,x0,y0,x1,y1,label,confidence\n");
    for (ImageRects& rects: itemsList)
    {
        flexible_nms(avgImages, rects.rects);
        for (const Rect &r: rects.rects)
        {
            if (!r.dropped)
            {
//                if (r.confidence > 0.001)
                printf("%s,%.1f,%.1f,%.1f,%.1f,car,%.3f\n",
                       rects.image_filename.c_str(),
                       r.x0, r.y0, r.x1, r.y1, r.confidence);

                savedRects++;
            }
        }
    }
    qDebug() << savedRects;
}
